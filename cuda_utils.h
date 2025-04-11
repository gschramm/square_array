#pragma once

#include <cuda_runtime.h>

template <typename T>
void handle_cuda_input_array(const T *host_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);
