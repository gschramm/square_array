#include "square_array.h"
#include "square_op.h"

void square_array(float* array, size_t size, float* result_sum, int device_id) {
    *result_sum = 0.0f;

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        compute_and_accumulate(array, i, result_sum);
    }
}

