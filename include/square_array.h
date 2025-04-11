#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void square_array(float* array, size_t size, float* result_sum, int device_id = 0);

#ifdef __cplusplus
}
#endif
