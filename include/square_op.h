#pragma once
#include "cuda_compat.h"
#include "utils.h"

CUDA_HOST_DEVICE inline void compute_and_accumulate(float* array, size_t idx, float* result_sum) {
    array[idx] += 1;
    atomic_sum(result_sum, array[idx]);
}

