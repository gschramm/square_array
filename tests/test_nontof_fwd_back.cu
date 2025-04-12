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

    cudaDeviceSynchronize();

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

    cudaFree(img_origin);
    cudaFree(img);
    cudaFree(vstart);
    cudaFree(vend);
    cudaFree(xstart);
    cudaFree(xend);
    cudaFree(img_fwd);
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

    cudaFree(img);
    cudaFree(vstart);
    cudaFree(vend);
    cudaFree(xstart);
    cudaFree(xend);
    cudaFree(img_fwd);
}
