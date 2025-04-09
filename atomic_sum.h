#pragma once
#include "cuda_compat.h"

WORKER_QUALIFIER inline void atomic_sum(float* target, float value) {
#ifdef __CUDA_ARCH__
    atomicAdd(target, value);
#else
#pragma omp atomic
    *target += value;
#endif
}
