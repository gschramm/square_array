#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

void test_host_arrays();
void test_cuda_managed_arrays();
void test_cuda_device_arrays();

int main()
{
    std::cout << "Testing joseph3d_fwd and joseph3d_back with different memory modes:\n";

    std::cout << "\n--- Testing with Host Arrays ---\n";
    test_host_arrays();

    std::cout << "\n--- Testing with CUDA-Managed Arrays ---\n";
    test_cuda_managed_arrays();

    std::cout << "\n--- Testing with CUDA Device Arrays ---\n";
    test_cuda_device_arrays();

    return 0;
}

void test_host_arrays()
{
    // Host array test (same as your current implementation)
    std::vector<int> img_dim = {2, 3, 4};
    std::vector<float> voxsize = {4.0f, 3.0f, 2.0f};

    std::vector<float> img_origin(3);
    for (int i = 0; i < 3; ++i)
    {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5f) * voxsize[i];
    }

    std::vector<float> img = readArrayFromFile<float>("img.txt");
    std::vector<float> vstart = readArrayFromFile<float>("vstart.txt");
    std::vector<float> vend = readArrayFromFile<float>("vend.txt");
    size_t nlors = vstart.size() / 3;

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

    std::vector<float> img_fwd(nlors);
    joseph3d_fwd(
        xstart.data(), xend.data(), img.data(),
        img_origin.data(), voxsize.data(), img_fwd.data(),
        nlors, img_dim.data(), 0, 64);

    std::vector<float> expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
    float fwd_diff = 0;
    float eps = 1e-7;

    for (int ir = 0; ir < nlors; ir++)
    {
        fwd_diff = std::abs(img_fwd[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            std::cerr << "Host array test failed for ray " << ir << "\n";
            return;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // Test the back projection using the defintion of the adjoint operator
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
        printf("\n#########################################################################");
        printf("\nback projection test failed. back projection seems not to be the adjoint.");
        printf("\n %.7e", ip_diff);
        printf("\n#########################################################################\n");
        std::cerr << "Back projection via adjointness test failed.\n";
    }
}

void test_cuda_managed_arrays()
{
    // CUDA-managed array test
    int img_dim[3] = {2, 3, 4};
    float voxsize[3] = {4.0f, 3.0f, 2.0f};

    float *img_origin;
    cudaMallocManaged(&img_origin, 3 * sizeof(float));
    for (int i = 0; i < 3; ++i)
    {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5f) * voxsize[i];
    }

    std::vector<float> img_from_file = readArrayFromFile<float>("img.txt");
    float *img;
    cudaMallocManaged(&img, img_from_file.size() * sizeof(float));
    std::copy(img_from_file.begin(), img_from_file.end(), img);

    std::vector<float> vstart_from_file = readArrayFromFile<float>("vstart.txt");
    float *vstart;
    cudaMallocManaged(&vstart, vstart_from_file.size() * sizeof(float));
    std::copy(vstart_from_file.begin(), vstart_from_file.end(), vstart);

    std::vector<float> vend_from_file = readArrayFromFile<float>("vend.txt");
    float *vend;
    cudaMallocManaged(&vend, vend_from_file.size() * sizeof(float));
    std::copy(vend_from_file.begin(), vend_from_file.end(), vend);

    size_t nlors = vstart_from_file.size() / 3;

    float *xstart, *xend;
    cudaMallocManaged(&xstart, 3 * nlors * sizeof(float));
    cudaMallocManaged(&xend, 3 * nlors * sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        for (int j = 0; j < 3; j++)
        {
            xstart[ir * 3 + j] = img_origin[j] + vstart[ir * 3 + j] * voxsize[j];
            xend[ir * 3 + j] = img_origin[j] + vend[ir * 3 + j] * voxsize[j];
        }
    }

    float *img_fwd;
    cudaMallocManaged(&img_fwd, nlors * sizeof(float));
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nlors, img_dim, 0, 64);

    std::vector<float> expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
    float fwd_diff = 0;
    float eps = 1e-7;

    for (int ir = 0; ir < nlors; ir++)
    {
        fwd_diff = std::abs(img_fwd[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            std::cerr << "CUDA-managed array test failed for ray " << ir << "\n";
            return;
        }
    }

    // Test the back projection
    float *bimg;
    cudaMallocManaged(&bimg, img_dim[0] * img_dim[1] * img_dim[2] * sizeof(float));
    std::fill(bimg, bimg + (img_dim[0] * img_dim[1] * img_dim[2]), 0.0f);

    float *ones;
    cudaMallocManaged(&ones, nlors * sizeof(float));
    std::fill(ones, ones + nlors, 1.0f);

    joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones, nlors, img_dim);

    printf("\nCUDA-managed back projection of ones along all rays:\n");
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

    // Validate the back projection using adjointness
    float inner_product1 = 0.0f;
    float inner_product2 = 0.0f;

    for (size_t i = 0; i < img_from_file.size(); i++)
    {
        inner_product1 += img[i] * bimg[i];
    }

    for (size_t ir = 0; ir < nlors; ir++)
    {
        inner_product2 += img_fwd[ir] * ones[ir];
    }

    float ip_diff = fabs(inner_product1 - inner_product2);

    if (ip_diff > eps)
    {
        std::cerr << "CUDA-managed array back projection test failed: adjointness property violated.\n";
    }
    else
    {
        std::cout << "CUDA-managed array back projection test passed.\n";
    }

    cudaFree(img_origin);
    cudaFree(img);
    cudaFree(vstart);
    cudaFree(vend);
    cudaFree(xstart);
    cudaFree(xend);
    cudaFree(img_fwd);
    cudaFree(bimg);
    cudaFree(ones);
}

void test_cuda_device_arrays()
{
    // CUDA device array test
    int img_dim[3] = {2, 3, 4};
    float voxsize[3] = {4.0f, 3.0f, 2.0f};

    float img_origin[3];
    for (int i = 0; i < 3; ++i)
    {
        img_origin[i] = (-(float)img_dim[i] / 2 + 0.5f) * voxsize[i];
    }

    std::vector<float> img_from_file = readArrayFromFile<float>("img.txt");
    float *img;
    cudaMalloc(&img, img_from_file.size() * sizeof(float));
    cudaMemcpy(img, img_from_file.data(), img_from_file.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> vstart_from_file = readArrayFromFile<float>("vstart.txt");
    float *vstart;
    cudaMalloc(&vstart, vstart_from_file.size() * sizeof(float));
    cudaMemcpy(vstart, vstart_from_file.data(), vstart_from_file.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> vend_from_file = readArrayFromFile<float>("vend.txt");
    float *vend;
    cudaMalloc(&vend, vend_from_file.size() * sizeof(float));
    cudaMemcpy(vend, vend_from_file.data(), vend_from_file.size() * sizeof(float), cudaMemcpyHostToDevice);

    size_t nlors = vstart_from_file.size() / 3;

    float *xstart, *xend;
    cudaMalloc(&xstart, 3 * nlors * sizeof(float));
    cudaMalloc(&xend, 3 * nlors * sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        for (int j = 0; j < 3; j++)
        {
            float xstart_val = img_origin[j] + vstart_from_file[ir * 3 + j] * voxsize[j];
            float xend_val = img_origin[j] + vend_from_file[ir * 3 + j] * voxsize[j];
            cudaMemcpy(&xstart[ir * 3 + j], &xstart_val, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(&xend[ir * 3 + j], &xend_val, sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    float *img_fwd;
    cudaMalloc(&img_fwd, nlors * sizeof(float));
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nlors, img_dim, 0, 64);

    std::vector<float> img_fwd_host(nlors);
    cudaMemcpy(img_fwd_host.data(), img_fwd, nlors * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
    float fwd_diff = 0;
    float eps = 1e-7;

    for (int ir = 0; ir < nlors; ir++)
    {
        fwd_diff = std::abs(img_fwd_host[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            std::cerr << "CUDA device array test failed for ray " << ir << "\n";
            return;
        }
    }

    // Test the back projection
    float *bimg;
    cudaMalloc(&bimg, img_dim[0] * img_dim[1] * img_dim[2] * sizeof(float));
    cudaMemset(bimg, 0, img_dim[0] * img_dim[1] * img_dim[2] * sizeof(float));

    float *ones;
    cudaMalloc(&ones, nlors * sizeof(float));
    cudaMemset(ones, 0, nlors * sizeof(float));
    std::vector<float> ones_host(nlors, 1.0f);
    cudaMemcpy(ones, ones_host.data(), nlors * sizeof(float), cudaMemcpyHostToDevice);

    joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones, nlors, img_dim);

    std::vector<float> bimg_host(img_dim[0] * img_dim[1] * img_dim[2]);
    cudaMemcpy(bimg_host.data(), bimg, bimg_host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nCUDA device back projection of ones along all rays:\n");
    for (size_t i0 = 0; i0 < img_dim[0]; i0++)
    {
        for (size_t i1 = 0; i1 < img_dim[1]; i1++)
        {
            for (size_t i2 = 0; i2 < img_dim[2]; i2++)
            {
                printf("%.1f ", bimg_host[img_dim[1] * img_dim[2] * i0 + img_dim[2] * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Validate the back projection using adjointness
    float inner_product1 = 0.0f;
    float inner_product2 = 0.0f;

    for (size_t i = 0; i < img_from_file.size(); i++)
    {
        inner_product1 += img_from_file[i] * bimg_host[i];
    }

    for (size_t ir = 0; ir < nlors; ir++)
    {
        inner_product2 += img_fwd_host[ir] * ones_host[ir];
    }

    float ip_diff = fabs(inner_product1 - inner_product2);

    if (ip_diff > eps)
    {
        std::cerr << "CUDA device array back projection test failed: adjointness property violated.\n";
    }
    else
    {
        std::cout << "CUDA device array back projection test passed.\n";
    }

    cudaFree(img);
    cudaFree(vstart);
    cudaFree(vend);
    cudaFree(xstart);
    cudaFree(xend);
    cudaFree(img_fwd);
    cudaFree(bimg);
    cudaFree(ones);
}
