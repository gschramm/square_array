#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstring>

int main()
{
    size_t nlors = 10;

    // get the number of cuda devices - because we want to run on the last device
    int device_count;
    cudaGetDeviceCount(&device_count);

    ////////////////////////////////////////////////////////
    // CUDA memory managed use case
    ////////////////////////////////////////////////////////

    std::cout << "CUDA managed memory use case\n";

    cudaSetDevice(device_count - 1);

    int *img_dim;
    cudaMallocManaged(&img_dim, 3 * sizeof(int));
    img_dim[0] = 2;
    img_dim[1] = 3;
    img_dim[2] = 4;

    float *voxsize;
    cudaMallocManaged(&voxsize, 3 * sizeof(float));
    voxsize[0] = 4;
    voxsize[1] = 3;
    voxsize[2] = 2;

    float *img_origin;
    cudaMallocManaged(&img_origin, 3 * sizeof(float));
    for (int i = 0; i < 3; ++i)
    {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5) * voxsize[i];
    }

    // read the image from file and convert into a cuda managed array
    std::vector<float> img_from_file = readArrayFromFile<float>("img.txt");
    float *img;
    cudaMallocManaged(&img, (img_dim[0] * img_dim[1] * img_dim[2]) * sizeof(float));
    memcpy(img, img_from_file.data(), img_from_file.size() * sizeof(float));

    // read the ray start coordinates in voxel coordinates
    // reast vstart from vstart.txt and convert into a cuda managed array
    std::vector<float> vstart_from_file = readArrayFromFile<float>("vstart.txt");
    float *vstart;
    cudaMallocManaged(&vstart, (3 * nlors) * sizeof(float));
    memcpy(vstart, vstart_from_file.data(), vstart_from_file.size() * sizeof(float));

    // read the ray end coordinates in voxel coordinates
    // read vend from vend.txt and convert into a cuda managed array
    std::vector<float> vend_from_file = readArrayFromFile<float>("vend.txt");
    float *vend;
    cudaMallocManaged(&vend, (3 * nlors) * sizeof(float));
    memcpy(vend, vend_from_file.data(), vend_from_file.size() * sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d\n", ir);
        printf("start voxel num .: %.1f %.1f %.1f\n", vstart[ir * 3 + 0], vstart[ir * 3 + 1], vstart[ir * 3 + 2]);
        printf("end   voxel num .: %.1f %.1f %.1f\n", vend[ir * 3 + 0], vend[ir * 3 + 1], vend[ir * 3 + 2]);
    }

    // calculate the start and end coordinates in world coordinates
    float *xstart;
    cudaMallocManaged(&xstart, (3 * nlors) * sizeof(float));
    float *xend;
    cudaMallocManaged(&xend, (3 * nlors) * sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        xstart[ir * 3 + 0] = img_origin[0] + vstart[ir * 3 + 0] * voxsize[0];
        xstart[ir * 3 + 1] = img_origin[1] + vstart[ir * 3 + 1] * voxsize[1];
        xstart[ir * 3 + 2] = img_origin[2] + vstart[ir * 3 + 2] * voxsize[2];

        xend[ir * 3 + 0] = img_origin[0] + vend[ir * 3 + 0] * voxsize[0];
        xend[ir * 3 + 1] = img_origin[1] + vend[ir * 3 + 1] * voxsize[1];
        xend[ir * 3 + 2] = img_origin[2] + vend[ir * 3 + 2] * voxsize[2];
    }

    float *img_fwd;
    cudaMallocManaged(&img_fwd, nlors * sizeof(float));

    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nlors, img_dim, 0, 64);

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    int retval = 0;
    float eps = 1e-7;

    // read the expected_fwd_vals from expected_fwd_vals.txt and convert into a cuda managed array
    std::vector<float> expected_fwd_vals_from_file = readArrayFromFile<float>("expected_fwd_vals.txt");
    float *expected_fwd_vals;
    cudaMallocManaged(&expected_fwd_vals, (nlors) * sizeof(float));
    memcpy(expected_fwd_vals, expected_fwd_vals_from_file.data(), expected_fwd_vals_from_file.size() * sizeof(float));

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

    // test the back projection

    float *bimg;
    cudaMallocManaged(&bimg, (img_dim[0] * img_dim[1] * img_dim[2]) * sizeof(float));

    for (size_t i = 0; i < (img_dim[0] * img_dim[1] * img_dim[2]); i++)
    {
        bimg[i] = 0;
    }

    float *ones;
    cudaMallocManaged(&ones, nlors * sizeof(float));
    for (size_t i = 0; i < nlors; i++)
    {
        ones[i] = 1;
    }

    joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones, nlors, img_dim, 0, 64);

    printf("\nback projection of ones along all rays:\n");
    for (size_t i0 = 0; i0 < img_dim[0]; i0++)
    {
        for (size_t i1 = 0; i1 < img_dim[1]; i1++)
        {
            for (size_t i2 = 0; i2 < img_dim[2]; i2++)
            {
                printf("%.1f ", bimg[img_dim[1] * img_dim[2] * i0 + img_dim[2] * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // To test whether the back projection is correct, we test if the back projector is the adjoint
    // of the forward projector. This is more practical than checking a lot of single voxels in the
    // back projected image.

    float inner_product1 = 0;
    float inner_product2 = 0;

    for (size_t i = 0; i < (img_dim[0] * img_dim[1] * img_dim[2]); i++)
    {
        inner_product1 += (img[i] * bimg[i]);
    }

    for (size_t ir = 0; ir < nlors; ir++)
    {
        inner_product2 += (img_fwd[ir] * ones[ir]);
    }

    float ip_diff = fabs(inner_product1 - inner_product2);

    if (ip_diff > eps)
    {
        printf("\n#########################################################################");
        printf("\nback projection test failed. back projection seems not to be the adjoint.");
        printf("\n %.7e", ip_diff);
        printf("\n#########################################################################\n");
        retval = 1;
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    cudaFree(img_dim);
    cudaFree(voxsize);
    cudaFree(img_origin);
    cudaFree(img);
    cudaFree(xstart);
    cudaFree(xend);
    cudaFree(img_fwd);

    cudaFree(bimg);
    cudaFree(ones);

    return retval;
}
