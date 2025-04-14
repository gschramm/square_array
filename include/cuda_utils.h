#pragma once

#include <cuda_runtime.h>

// Overload for constant input_ptr (const T*)
template <typename T>
void handle_cuda_input_array(const T *input_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);

// Overload for non-constant input_ptr (T*)
template <typename T>
void handle_cuda_input_array(T *input_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);
