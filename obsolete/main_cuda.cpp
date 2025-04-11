#include "square_array.h"
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

void print_array(const char* label, float* array, size_t size) {
    std::cout << label << ": ";
    // print max 10 elements
    size_t print_size = (size > 10) ? 10 : size;
    for (size_t i = 0; i < print_size; ++i)
        std::cout << array[i] << " ";
    // print ellipses if size > 10 and the last element
    if (size > 10)
        std::cout << "... " << array[size - 1];
    std::cout << "\n";
}

int main() {
    const size_t repetitions = 15;
    const size_t N = 1'000'000;
    float sum = 0.0f;

    //////////////////////////////////////////////////////
    // Host array use case
    //////////////////////////////////////////////////////

    // get the number of cuda devices - because we want to run on the last device
    int device_count;
    cudaGetDeviceCount(&device_count);

    float* host_data = new float[N];
    for (size_t i = 0; i < N; ++i)
        host_data[i] = i + 1;

    std::cout << "\n";
    // print_array("Host input", host_data, N);

    try { 
        // use invalid device ID 999 for testing
        square_array(host_data, N, &sum, 999);
        print_array("Host output", host_data, N);
        std::cout << "Host sum of squares: " << sum << "\n";
    }
    catch (const std::runtime_error& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < repetitions; i++)
    {
        // use default device ID 0
        square_array(host_data, N, &sum, device_count - 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    print_array("Host output", host_data, N);
    std::cout << "Host output sum: " << sum << "\n";
    std::cout << "Host operation took " << duration / 1e3 << " ms\n";

    delete[] host_data;

    ////////////////////////////////////////////////////////
    // Device array use case
    ////////////////////////////////////////////////////////

    float sum_d = 0.0f;
    float* temp = new float[N];
    for (size_t i = 0; i < N; ++i)
        temp[i] = i + 1;
    float* dev_data;
    cudaSetDevice(0);
    cudaMalloc(&dev_data, N * sizeof(float));
    cudaMemcpy(dev_data, temp, N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "\n";
    // print_array("Device input", temp, N);

    start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < repetitions; i++)
    {
        // if we are passing a device array, the device ID is ignored
    // because the array is already on the device
    // the device ID is obtained from the device array itself
    square_array(dev_data, N, &sum_d, 999);
    }
    cudaMemcpy(temp, dev_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    print_array("Device output", temp, N);
    std::cout << "Device output sum: " << sum_d << "\n";
    std::cout << "Device operation took " << duration / 1e3 << " ms\n";
    cudaFree(dev_data);

    delete[] temp;

    ////////////////////////////////////////////////////////
    // CUDA memory managed use case
    ////////////////////////////////////////////////////////

    float sum_m = 0.0f;
    float* managed;
    cudaSetDevice(device_count - 1);
    cudaMallocManaged(&managed, N * sizeof(float));
    for (size_t i = 0; i < N; ++i)
        managed[i] = i + 1;

    std::cout << "\n";
    // print_array("Managed input", managed, N);

    // if we are passing a cuda managed array, the device ID is ignored
    // because the array is already on the device
    start = std::chrono::high_resolution_clock::now();
    // the device ID is obtained from the device array itself
    square_array(managed, N, &sum_m, 999);
    end = std::chrono::high_resolution_clock::now();
    auto first_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    for (auto i = 0; i < repetitions - 1; i++)
    {
        square_array(managed, N, &sum_m, device_count - 1);
    }
    cudaMemcpy(temp, dev_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    print_array("Device output", managed, N);
    std::cout << "Device output sum: " << sum_m << "\n";
    std::cout << "Device operation took " << (first_duration + duration) / 1e3 << " ms (" << first_duration / 1e3 << " vs. " << duration / (repetitions - 1) / 1e3 << ")\n";
    cudaDeviceSynchronize();
    cudaFree(managed);

    return 0;
}

