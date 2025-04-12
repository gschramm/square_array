#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

int main()
{
    ////////////////////////////////////////////////////////
    // OpenMP managed memory use case
    ////////////////////////////////////////////////////////

    std::cout << "OpenMP use case\n";

    int img_dim[3] = {2, 3, 4};
    float voxsize[3] = {4, 3, 2};

    float img_origin[3];
    for (int i = 0; i < 3; ++i)
    {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5) * voxsize[i];
    }

    // Read the image from file
    std::vector<float> img_from_file = readArrayFromFile<float>("img.txt");

    // Read the ray start coordinates from file
    std::vector<float> vstart_from_file = readArrayFromFile<float>("vstart.txt");

    // Read the ray end coordinates from file
    std::vector<float> vend_from_file = readArrayFromFile<float>("vend.txt");

    size_t nlors = vstart_from_file.size() / 3;

    // Calculate the start and end coordinates in world coordinates
    std::vector<float> xstart(3 * nlors);
    std::vector<float> xend(3 * nlors);

    for (int ir = 0; ir < nlors; ir++)
    {
        xstart[ir * 3 + 0] = img_origin[0] + vstart_from_file[ir * 3 + 0] * voxsize[0];
        xstart[ir * 3 + 1] = img_origin[1] + vstart_from_file[ir * 3 + 1] * voxsize[1];
        xstart[ir * 3 + 2] = img_origin[2] + vstart_from_file[ir * 3 + 2] * voxsize[2];

        xend[ir * 3 + 0] = img_origin[0] + vend_from_file[ir * 3 + 0] * voxsize[0];
        xend[ir * 3 + 1] = img_origin[1] + vend_from_file[ir * 3 + 1] * voxsize[1];
        xend[ir * 3 + 2] = img_origin[2] + vend_from_file[ir * 3 + 2] * voxsize[2];
    }

    // Allocate memory for forward projection results
    std::vector<float> img_fwd(nlors);

    // Perform forward projection
    joseph3d_fwd(
        xstart.data(), xend.data(), img_from_file.data(),
        img_origin, voxsize, img_fwd.data(),
        nlors, img_dim, 0, 64);

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    // Read the expected forward values from file
    std::vector<float> expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");

    // Check if we got the expected results
    float fwd_diff = 0;
    float eps = 1e-7;
    int retval = 0;

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

    // Test the back projection
    std::vector<float> bimg(img_dim[0] * img_dim[1] * img_dim[2], 0.0f);
    std::vector<float> ones(nlors, 1.0f);

    joseph3d_back(
        xstart.data(), xend.data(), bimg.data(),
        img_origin, voxsize, ones.data(),
        nlors, img_dim);

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
        inner_product1 += (img_from_file[i] * bimg[i]);
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

    return retval;
}
