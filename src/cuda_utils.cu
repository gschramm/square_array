#include "cuda_utils.h"
#include <iostream>

// Overload for constant input_ptr (const T*)
template <typename T>
void handle_cuda_input_array(const T *input_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, input_ptr);
    free_flag = false;

    if (err == cudaSuccess && attr.type == cudaMemoryTypeManaged)
    {
        // Prefetch and advise for managed memory
        cudaMemPrefetchAsync(const_cast<void *>(static_cast<const void *>(input_ptr)), size, device_id);
        cudaMemAdvise(const_cast<void *>(static_cast<const void *>(input_ptr)), size, memory_hint, device_id);
    }

    if (err == cudaSuccess && (attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice))
    {
        // Assign managed or device pointer
        *device_ptr = const_cast<T *>(input_ptr);
    }
    else
    {
        // Host pointer case, transfer to device
        cudaMalloc(device_ptr, size);
        cudaMemcpy(*device_ptr, input_ptr, size, cudaMemcpyHostToDevice);
        free_flag = true;
    }
}

// Overload for non-constant input_ptr (T*) (not const)
template <typename T>
void handle_cuda_input_array(T *input_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, input_ptr);
    free_flag = false;

    if (err == cudaSuccess && attr.type == cudaMemoryTypeManaged)
    {
        // Prefetch and advise for managed memory
        cudaMemPrefetchAsync(input_ptr, size, device_id);
        cudaMemAdvise(input_ptr, size, memory_hint, device_id);
    }

    if (err == cudaSuccess && (attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice))
    {
        // Assign managed or device pointer
        *device_ptr = input_ptr;
    }
    else
    {
        // Host pointer case, transfer to device
        cudaMalloc(device_ptr, size);
        cudaMemcpy(*device_ptr, input_ptr, size, cudaMemcpyHostToDevice);
        free_flag = true;
    }
}

// Explicit template instantiations
template void handle_cuda_input_array<double>(const double *, double **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<double>(double *, double **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<float>(const float *, float **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<float>(float *, float **, size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<int>(const int *, int **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<int>(int *, int **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned int>(const unsigned int *, unsigned int **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned int>(unsigned int *, unsigned int **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<size_t>(const size_t *, size_t **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<size_t>(size_t *, size_t **, size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<char>(const char *, char **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<char>(char *, char **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned char>(const unsigned char *, unsigned char **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned char>(unsigned char *, unsigned char **, size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<bool>(const bool *, bool **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<bool>(bool *, bool **, size_t, bool &, int, cudaMemoryAdvise);
