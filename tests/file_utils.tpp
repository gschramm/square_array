#include <fstream>
#include <iostream>
#include <vector>

// Function to save a 1D array to a text file
template <typename T>
void saveArrayToFile(const std::string &filename, const T *array, size_t size)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    for (size_t i = 0; i < size; ++i)
    {
        file << array[i] << "\n";
    }
    file.close();
}

// Function to read a 1D array from a text file
template <typename T>
std::vector<T> readArrayFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for reading.\n";
        return {};
    }
    std::vector<T> array;
    T value;
    while (file >> value)
    {
        array.push_back(value);
    }
    file.close();
    return array;
}
