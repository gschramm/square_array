#pragma once

#include <string>
#include <vector>

// Function to save a 1D array to a text file
template <typename T>
void saveArrayToFile(const std::string &filename, const T *array, size_t size);

// Function to read a 1D array from a text file
template <typename T>
std::vector<T> readArrayFromFile(const std::string &filename);

#include "file_utils.tpp" // Include the template implementation
