#include "parallelproj.h"
#include <iostream>
#include <chrono>
#include <cmath>

int main() {
    const size_t repetitions = 5;
    size_t nlors = 10;

    ////////////////////////////////////////////////////////
    // OpenMP managed memory use case
    ////////////////////////////////////////////////////////

    std::cout << "OpenMP use case\n";

    int img_dim[3] = {2, 3, 4};

    float voxsize[3] = {4, 3, 2};

    float img_origin[3];
    for (int i = 0; i < 3; ++i) {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5) * voxsize[i];
    }

    float* img = new float[img_dim[0] * img_dim[1] * img_dim[2]];

    // fill the test image
    for (int i0 = 0; i0 < img_dim[0]; i0++)
    {
        for (int i1 = 0; i1 < img_dim[1]; i1++)
        {
            for (int i2 = 0; i2 < img_dim[2]; i2++)
            {
                img[img_dim[1] * img_dim[2] * i0 + img_dim[2] * i1 + i2] = float(img_dim[1] * img_dim[2] * i0 + img_dim[2] * i1 + i2 + 1);
                printf("%.1f ", img[img_dim[1] * img_dim[2] * i0 + img_dim[2] * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    float id0 = static_cast<float>(img_dim[0]);
    float id1 = static_cast<float>(img_dim[1]);
    float id2 = static_cast<float>(img_dim[2]);

    float vstart[] = {
        0, -1, 0,           // 0
        0, -1, 0,           // 1
        0, -1, 1,           // 2
        0, -1, 0.5,         // 3
        0, 0, -1,           // 4
        -1, 0, 0,           // 5
        id0 - 1, -1, 0,      // 6 - (shifted 1)
        id0 - 1, -1, id2 - 1, // 7 - (shifted 6)
        id0 - 1, 0, -1,      // 8 - (shifted 4)
        id0 - 1, id1 - 1, -1, // 9 - (shifted 8)
    };

    float vend[] = {
        0, id1, 0,           // 0
        0, id1, 0,           // 1
        0, id1, 1,           // 2
        0, id1, 0.5,         // 3
        0, 0, id2,           // 4
        id0, 0, 0,           // 5
        id0 - 1, id1, 0,      // 6 - (shifted 1)
        id0 - 1, id1, id2 - 1, // 7 - (shifted 6)
        id0 - 1, 0, id2,      // 8 - (shifted 4)
        id0 - 1, id1 - 1, id2, // 9 - (shifted 8)
    };

    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d\n", ir);
        printf("start voxel num .: %.1f %.1f %.1f\n", vstart[ir * 3 + 0], vstart[ir * 3 + 1], vstart[ir * 3 + 2]);
        printf("end   voxel num .: %.1f %.1f %.1f\n", vend[ir * 3 + 0], vend[ir * 3 + 1], vend[ir * 3 + 2]);
    }

    // calculate the start and end coordinates in world coordinates
    
    float *xstart = new float[3 * nlors];
    float *xend = new float[3 * nlors];

    for (int ir = 0; ir < nlors; ir++)

    {
        for (int j = 0; j < 3; j++)
        {
            xstart[ir * 3 + j] = img_origin[j] + vstart[ir * 3 + j] * voxsize[j];
            xend[ir * 3 + j] = img_origin[j] + vend[ir * 3 + j] * voxsize[j];
        }
    }

    float *img_fwd = new float[nlors];
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nlors, img_dim, 0, 64);

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    // calculate the expected values

    int retval = 0;
    float eps = 1e-7;

    float* expected_fwd_vals = new float[nlors];
    // initialize expected_fwd_vals with 0s
    for (int ir = 0; ir < nlors; ir++)
    {
        expected_fwd_vals[ir] = 0;
    }

    for (int i1 = 0; i1 < img_dim[1]; i1++)
    {
        expected_fwd_vals[0] += img[0 * img_dim[1] * img_dim[2] + i1 * img_dim[2] + 0] * voxsize[1];
    }

    expected_fwd_vals[1] = expected_fwd_vals[0];

    // calculate the expected value of ray2 from [0,-1,1] to [0,last+1,1]
    for (int i1 = 0; i1 < img_dim[1]; i1++)
    {
        expected_fwd_vals[2] += img[0 * img_dim[1] * img_dim[2] + i1 * img_dim[2] + 1] * voxsize[1];
    }

    // calculate the expected value of ray3 from [0,-1,0.5] to [0,last+1,0.5]
    expected_fwd_vals[3] = 0.5 * (expected_fwd_vals[0] + expected_fwd_vals[2]);

    // calculate the expected value of ray4 from [0,0,-1] to [0,0,last+1]
    for (int i2 = 0; i2 < img_dim[2]; i2++)
    {
        expected_fwd_vals[4] += img[0 * img_dim[1] * img_dim[2] + 0 * img_dim[2] + i2] * voxsize[2];
    }

    // calculate the expected value of ray5 from [-1,0,0] to [last+1,0,0]
    for (int i0 = 0; i0 < img_dim[0]; i0++)
    {
        expected_fwd_vals[5] += img[i0 * img_dim[1] * img_dim[2] + 0 * img_dim[2] + 0] * voxsize[0];
    }

    // calculate the expected value of rays6 from [img_dim[0]-1,-1,0] to [img_dim[0]-1,last+1,0]
    for (int i1 = 0; i1 < img_dim[1]; i1++)
    {
        expected_fwd_vals[6] += img[(img_dim[0] - 1) * img_dim[1] * img_dim[2] + i1 * img_dim[2] + 0] * voxsize[1];
    }

    // calculate the expected value of rays7 from [img_dim[0]-1,-1,img_dim[2]-1] to [img_dim[0]-1,last+1,img_dim[2]-1]
    for (int i1 = 0; i1 < img_dim[1]; i1++)
    {
        expected_fwd_vals[7] += img[(img_dim[0] - 1) * img_dim[1] * img_dim[2] + i1 * img_dim[2] + (img_dim[2] - 1)] * voxsize[1];
    }

    // calculate the expected value of ray4 from [img_dim[0]-1,0,-1] to [img_dim[0]-1,0,last+1]
    for (int i2 = 0; i2 < img_dim[2]; i2++)
    {
        expected_fwd_vals[8] += img[(img_dim[0] - 1) * img_dim[1] * img_dim[2] + 0 * img_dim[2] + i2] * voxsize[2];
    }

    // calculate the expected value of ray4 from [img_dim[0]-1,0,-1] to [img_dim[0]-1,0,last+1]
    for (int i2 = 0; i2 < img_dim[2]; i2++)
    {
        expected_fwd_vals[9] += img[(img_dim[0] - 1) * img_dim[1] * img_dim[2] + (img_dim[1] - 1) * img_dim[2] + i2] * voxsize[2];
    }

    // check if we got the expected results
    float fwd_diff = 0;
    printf("\nforward projection test\n");
    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d: fwd projected: %.7e expected: %.7e\n", ir, img_fwd[ir], expected_fwd_vals[ir]);

        fwd_diff = std::abs(img_fwd[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            printf("\n################################################################################");
            printf("\nabs(fwd projected - expected value) = %.2e for ray%d above tolerance %.2e", fwd_diff, ir, eps);
            printf("\n################################################################################\n");
            retval = 1;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    free(img);
    free(xstart);
    free(xend);
    free(img_fwd);
    free(expected_fwd_vals);

    return retval;
}

