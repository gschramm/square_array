#include "square_array.h"
#include <iostream>

void print_array(const char* label, float* array, size_t size) {
    std::cout << label << ": ";
    for (size_t i = 0; i < size; ++i)
        std::cout << array[i] << " ";
    std::cout << "\n";
}

int main() {
    const size_t N = 10;
    float data[N], sum = 0.0f;

    for (size_t i = 0; i < N; ++i)
        data[i] = i + 1;

    print_array("Input", data, N);
    square_array(data, N, &sum);
    print_array("Output ", data, N);
    std::cout << "Sum of squares: " << sum << "\n";
    return 0;
}

