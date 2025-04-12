#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>

int main()
{
    ////////////////////////////////////////////////////////
    // host array test cases
    ////////////////////////////////////////////////////////

#ifdef __CUDACC__
    std::cout << "CUDA host array test\n";
#else
    std::cout << "OpenMP test\n";
#endif

    std::vector<int> img_dim = {2, 3, 4};
    std::vector<float> voxsize = {4.0f, 3.0f, 2.0f};

    std::vector<float> img_origin(3);
    for (int i = 0; i < 3; ++i)
    {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5f) * voxsize[i];
    }

    // Read the image from file
    std::vector<float> img = readArrayFromFile<float>("img.txt");

    // Read the ray start coordinates from file
    std::vector<float> vstart = readArrayFromFile<float>("vstart.txt");

    // Read the ray end coordinates from file
    std::vector<float> vend = readArrayFromFile<float>("vend.txt");

    size_t nlors = vstart.size() / 3;

    // Calculate the start and end coordinates in world coordinates
    std::vector<float> xstart(3 * nlors);
    std::vector<float> xend(3 * nlors);

    for (int ir = 0; ir < nlors; ir++)
    {
        xstart[ir * 3 + 0] = img_origin[0] + vstart[ir * 3 + 0] * voxsize[0];
        xstart[ir * 3 + 1] = img_origin[1] + vstart[ir * 3 + 1] * voxsize[1];
        xstart[ir * 3 + 2] = img_origin[2] + vstart[ir * 3 + 2] * voxsize[2];

        xend[ir * 3 + 0] = img_origin[0] + vend[ir * 3 + 0] * voxsize[0];
        xend[ir * 3 + 1] = img_origin[1] + vend[ir * 3 + 1] * voxsize[1];
        xend[ir * 3 + 2] = img_origin[2] + vend[ir * 3 + 2] * voxsize[2];
    }

    // Allocate memory for forward projection results
    std::vector<float> img_fwd(nlors);

    // Perform forward projection
    joseph3d_fwd(
        xstart.data(), xend.data(), img.data(),
        img_origin.data(), voxsize.data(), img_fwd.data(),
        nlors, img_dim.data(), 0, 64);

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    // Read the expected forward values from file
    std::vector<float> expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");

    // Check if we got the expected results
    float fwd_diff = 0;
    float eps = 1e-7;

    printf("\nforward projection test\n");
    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d: fwd projected: %.7e expected: %.7e\n", ir, img_fwd[ir], expected_fwd_vals[ir]);

        fwd_diff = std::abs(img_fwd[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            std::cerr << "Forward projection test failed.\n";
            std::cerr << "Difference: " << fwd_diff << "\n";
            std::cerr << "Expected: " << expected_fwd_vals[ir] << "\n";
            std::cerr << "Actual: " << img_fwd[ir] << "\n";
            std::cerr << "Tolerance: " << eps << "\n";
            std::cerr << "Ray index: " << ir << "\n";
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
        img_origin.data(), voxsize.data(), ones.data(),
        nlors, img_dim.data());

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

    float inner_product1 = std::inner_product(img.begin(), img.end(), bimg.begin(), 0.0f);
    float inner_product2 = std::inner_product(img_fwd.begin(), img_fwd.end(), ones.begin(), 0.0f);

    float ip_diff = fabs(inner_product1 - inner_product2);

    if (ip_diff > eps)
    {
        std::cerr << "Back projection test failed.\n";
        std::cerr << "Inner product 1: " << inner_product1 << "\n";
        std::cerr << "Inner product 2: " << inner_product2 << "\n";
        std::cerr << "Difference: " << ip_diff << "\n";
        std::cerr << "Tolerance: " << eps << "\n";
    }

    return 0;
}
