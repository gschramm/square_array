#include "cuda_utils.h"

template <typename T>
void handle_cuda_input_array(const T *host_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, host_ptr);
    free_flag = false;

    if (err == cudaSuccess && attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(const_cast<void *>(static_cast<const void *>(host_ptr)), size, device_id);
        cudaMemAdvise(const_cast<void *>(static_cast<const void *>(host_ptr)), size, memory_hint, device_id);
    }

    if (err == cudaSuccess && (attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice))
    {
        *device_ptr = const_cast<T *>(host_ptr);
    }
    else
    {
        // Host pointer case, transfer to device
        cudaMalloc(device_ptr, size);
        cudaMemcpy(*device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
        free_flag = true;
    }
}

// Explicit template instantiations
template void handle_cuda_input_array<float>(const float *, float **, size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<int>(const int *, int **, size_t, bool &, int, cudaMemoryAdvise);
