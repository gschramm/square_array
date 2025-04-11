#include "square_array.h"
#include "square_op.h"
#include "debug.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void square_kernel(float* data, size_t n, float* sum) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        compute_and_accumulate(data, idx, sum);
    }
}

void square_array(float* array, size_t size, float* result_sum, int device_id) {

    float* device_array = nullptr;
    float* device_sum = nullptr;
    bool needs_copy_back = false;
    float zero = 0.0f;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, array);

    bool is_device_ptr = false;

//#if CUDART_VERSION >= 10000
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)){
        is_device_ptr = true;
        DEBUG_PRINT("Array is on device: %d\n", attr.device);
    }
//#else
//    if (err == cudaSuccess && attr.memoryType == cudaMemoryTypeDevice)
//        is_device_ptr = true;
//#endif

    if (is_device_ptr) {
        device_array = array;
        // use the device ID of the pointer to set the device
        cudaSetDevice(attr.device);
    } else {

        // set the device ID as specified
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_id >= device_count) {
            throw std::runtime_error("Invalid CUDA device ID: " + std::to_string(device_id));
        }

        DEBUG_PRINT("Copying array from host to device\n");
        cudaSetDevice(device_id);

        cudaMalloc(&device_array, size * sizeof(float));
        cudaMemcpy(device_array, array, size * sizeof(float), cudaMemcpyHostToDevice);
        needs_copy_back = true;
    }

    // get and print the current cuda device ID
    int current_device_id;
    cudaGetDevice(&current_device_id);
    DEBUG_PRINT("Using CUDA device: %d\n", current_device_id);

    // Allocate memory for the sum on the device
    cudaMalloc(&device_sum, sizeof(float));
    cudaMemcpy(device_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    square_kernel<<<blocks, threads>>>(device_array, size, device_sum);
    cudaDeviceSynchronize();

    if (needs_copy_back) {
        cudaMemcpy(array, device_array, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_array);
    }

    cudaMemcpy(result_sum, device_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_sum);
}

