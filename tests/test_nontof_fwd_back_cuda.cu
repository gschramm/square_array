#include "parallelproj.h"
#include "file_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

void test_cuda_managed_arrays();
void test_cuda_device_arrays();

int main()
{
    std::cout << "\n--- Testing with CUDA-Managed Arrays ---\n";
    test_cuda_managed_arrays();

    std::cout << "\n--- Testing with CUDA Device Arrays ---\n";
    test_cuda_device_arrays();

    return 0;
}

void test_cuda_managed_arrays()
{
    // CUDA-managed array test
    int h_img_dim[3] = {2, 3, 4};
    float h_voxsize[3] = {4.0f, 3.0f, 2.0f};

    float *cm_img_origin;
    cudaMallocManaged(&cm_img_origin, 3 * sizeof(float));
    for (int i = 0; i < 3; ++i)
    {
        cm_img_origin[i] = (-(float)h_img_dim[i] / 2 + 0.5f) * h_voxsize[i];
    }

    std::vector<float> h_img = readArrayFromFile<float>("img.txt");
    float *cm_img;
    cudaMallocManaged(&cm_img, h_img.size() * sizeof(float));
    std::copy(h_img.begin(), h_img.end(), cm_img);

    std::vector<float> h_vstart = readArrayFromFile<float>("vstart.txt");
    float *cm_vstart;
    cudaMallocManaged(&cm_vstart, h_vstart.size() * sizeof(float));
    std::copy(h_vstart.begin(), h_vstart.end(), cm_vstart);

    std::vector<float> h_vend = readArrayFromFile<float>("vend.txt");
    float *cm_vend;
    cudaMallocManaged(&cm_vend, h_vend.size() * sizeof(float));
    std::copy(h_vend.begin(), h_vend.end(), cm_vend);

    size_t nlors = h_vstart.size() / 3;

    float *cm_xstart, *cm_xend;
    cudaMallocManaged(&cm_xstart, 3 * nlors * sizeof(float));
    cudaMallocManaged(&cm_xend, 3 * nlors * sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        for (int j = 0; j < 3; j++)
        {
            cm_xstart[ir * 3 + j] = cm_img_origin[j] + cm_vstart[ir * 3 + j] * h_voxsize[j];
            cm_xend[ir * 3 + j] = cm_img_origin[j] + cm_vend[ir * 3 + j] * h_voxsize[j];
        }
    }

    float *cm_img_fwd;
    cudaMallocManaged(&cm_img_fwd, nlors * sizeof(float));
    joseph3d_fwd(cm_xstart, cm_xend, cm_img, cm_img_origin, h_voxsize, cm_img_fwd, nlors, h_img_dim, 0, 64);

    std::vector<float> h_expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
    float fwd_diff = 0;
    float eps = 1e-7;

    for (int ir = 0; ir < nlors; ir++)
    {
        fwd_diff = std::abs(cm_img_fwd[ir] - h_expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            std::cerr << "CUDA-managed array test failed for ray " << ir << "\n";
            return;
        }
    }

    // Test the back projection
    float *cm_bimg;
    cudaMallocManaged(&cm_bimg, h_img_dim[0] * h_img_dim[1] * h_img_dim[2] * sizeof(float));
    std::fill(cm_bimg, cm_bimg + (h_img_dim[0] * h_img_dim[1] * h_img_dim[2]), 0.0f);

    float *cm_ones;
    cudaMallocManaged(&cm_ones, nlors * sizeof(float));
    std::fill(cm_ones, cm_ones + nlors, 1.0f);

    joseph3d_back(cm_xstart, cm_xend, cm_bimg, cm_img_origin, h_voxsize, cm_ones, nlors, h_img_dim);

    printf("\nCUDA-managed back projection of ones along all rays:\n");
    for (size_t i0 = 0; i0 < h_img_dim[0]; i0++)
    {
        for (size_t i1 = 0; i1 < h_img_dim[1]; i1++)
        {
            for (size_t i2 = 0; i2 < h_img_dim[2]; i2++)
            {
                printf("%.1f ", cm_bimg[h_img_dim[1] * h_img_dim[2] * i0 + h_img_dim[2] * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Validate the back projection using adjointness
    float inner_product1 = 0.0f;
    float inner_product2 = 0.0f;

    for (size_t i = 0; i < h_img.size(); i++)
    {
        inner_product1 += cm_img[i] * cm_bimg[i];
    }

    for (size_t ir = 0; ir < nlors; ir++)
    {
        inner_product2 += cm_img_fwd[ir] * cm_ones[ir];
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

    cudaFree(cm_img_origin);
    cudaFree(cm_img);
    cudaFree(cm_vstart);
    cudaFree(cm_vend);
    cudaFree(cm_xstart);
    cudaFree(cm_xend);
    cudaFree(cm_img_fwd);
    cudaFree(cm_bimg);
    cudaFree(cm_ones);
}

void test_cuda_device_arrays()
{
    // CUDA device array test
    int h_img_dim[3] = {2, 3, 4};
    float h_voxsize[3] = {4.0f, 3.0f, 2.0f};

    float h_img_origin[3];
    for (int i = 0; i < 3; ++i)
    {
        h_img_origin[i] = (-(float)h_img_dim[i] / 2 + 0.5f) * h_voxsize[i];
    }

    std::vector<float> h_img = readArrayFromFile<float>("img.txt");
    float *d_img;
    cudaMalloc(&d_img, h_img.size() * sizeof(float));
    cudaMemcpy(d_img, h_img.data(), h_img.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_vstart = readArrayFromFile<float>("vstart.txt");
    float *d_vstart;
    cudaMalloc(&d_vstart, h_vstart.size() * sizeof(float));
    cudaMemcpy(d_vstart, h_vstart.data(), h_vstart.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_vend = readArrayFromFile<float>("vend.txt");
    float *d_vend;
    cudaMalloc(&d_vend, h_vend.size() * sizeof(float));
    cudaMemcpy(d_vend, h_vend.data(), h_vend.size() * sizeof(float), cudaMemcpyHostToDevice);

    size_t nlors = h_vstart.size() / 3;

    float *d_xstart, *d_xend;
    cudaMalloc(&d_xstart, 3 * nlors * sizeof(float));
    cudaMalloc(&d_xend, 3 * nlors * sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        for (int j = 0; j < 3; j++)
        {
            float xstart_val = h_img_origin[j] + h_vstart[ir * 3 + j] * h_voxsize[j];
            float xend_val = h_img_origin[j] + h_vend[ir * 3 + j] * h_voxsize[j];
            cudaMemcpy(&d_xstart[ir * 3 + j], &xstart_val, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_xend[ir * 3 + j], &xend_val, sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    float *d_img_fwd;
    cudaMalloc(&d_img_fwd, nlors * sizeof(float));
    joseph3d_fwd(d_xstart, d_xend, d_img, h_img_origin, h_voxsize, d_img_fwd, nlors, h_img_dim, 0, 64);

    std::vector<float> h_img_fwd(nlors);
    cudaMemcpy(h_img_fwd.data(), d_img_fwd, nlors * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> h_expected_fwd_vals = readArrayFromFile<float>("expected_fwd_vals.txt");
    float fwd_diff = 0;
    float eps = 1e-7;

    for (int ir = 0; ir < nlors; ir++)
    {
        fwd_diff = std::abs(h_img_fwd[ir] - h_expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            std::cerr << "CUDA device array test failed for ray " << ir << "\n";
            return;
        }
    }

    // Test the back projection
    float *d_bimg;
    cudaMalloc(&d_bimg, h_img_dim[0] * h_img_dim[1] * h_img_dim[2] * sizeof(float));
    cudaMemset(d_bimg, 0, h_img_dim[0] * h_img_dim[1] * h_img_dim[2] * sizeof(float));

    float *d_ones;
    cudaMalloc(&d_ones, nlors * sizeof(float));
    std::vector<float> h_ones(nlors, 1.0f);
    cudaMemcpy(d_ones, h_ones.data(), nlors * sizeof(float), cudaMemcpyHostToDevice);

    joseph3d_back(d_xstart, d_xend, d_bimg, h_img_origin, h_voxsize, d_ones, nlors, h_img_dim);

    std::vector<float> h_bimg(h_img_dim[0] * h_img_dim[1] * h_img_dim[2]);
    cudaMemcpy(h_bimg.data(), d_bimg, h_bimg.size() * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nCUDA device back projection of ones along all rays:\n");
    for (size_t i0 = 0; i0 < h_img_dim[0]; i0++)
    {
        for (size_t i1 = 0; i1 < h_img_dim[1]; i1++)
        {
            for (size_t i2 = 0; i2 < h_img_dim[2]; i2++)
            {
                printf("%.1f ", h_bimg[h_img_dim[1] * h_img_dim[2] * i0 + h_img_dim[2] * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Validate the back projection using adjointness
    float inner_product1 = 0.0f;
    float inner_product2 = 0.0f;

    for (size_t i = 0; i < h_img.size(); i++)
    {
        inner_product1 += h_img[i] * h_bimg[i];
    }

    for (size_t ir = 0; ir < nlors; ir++)
    {
        inner_product2 += h_img_fwd[ir] * h_ones[ir];
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

    cudaFree(d_img);
    cudaFree(d_vstart);
    cudaFree(d_vend);
    cudaFree(d_xstart);
    cudaFree(d_xend);
    cudaFree(d_img_fwd);
    cudaFree(d_bimg);
    cudaFree(d_ones);
}
