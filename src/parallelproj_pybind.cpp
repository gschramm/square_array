#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>
#include "parallelproj.h"

namespace py = pybind11;

// Helper function to extract raw pointer and shape
template <typename T>
std::pair<T *, std::vector<size_t>> extract_pointer_and_shape(py::object array)
{
    T *raw_ptr = nullptr;
    std::vector<size_t> shape;

    // Handle arrays using the __dlpack__ protocol (default)
    if (py::hasattr(array, "__dlpack__"))
    {
        // Call the __dlpack__ method to get a DLPack tensor
        py::capsule dlpack_capsule = array.attr("__dlpack__")();

        // Extract the DLManagedTensor from the capsule
        auto *managed_tensor = static_cast<DLManagedTensor *>(
            PyCapsule_GetPointer(dlpack_capsule.ptr(), "dltensor"));

        if (!managed_tensor)
        {
            throw std::runtime_error("Failed to extract DLManagedTensor from PyCapsule.");
        }

        // Access the DLTensor from the DLManagedTensor
        DLTensor dltensor = managed_tensor->dl_tensor;

        // Ensure the data type matches
        if (dltensor.dtype.code != kDLFloat || dltensor.dtype.bits != sizeof(T) * 8)
        {
            throw std::invalid_argument("DLPack tensor has an incompatible data type.");
        }

        // Get the raw pointer and shape
        raw_ptr = reinterpret_cast<T *>(dltensor.data);
        shape = std::vector<size_t>(dltensor.shape, dltensor.shape + dltensor.ndim);
    }
    // Handle NumPy arrays
    else if (py::isinstance<py::array_t<T>>(array))
    {
        auto numpy_array = array.cast<py::array_t<T>>();
        raw_ptr = numpy_array.mutable_data();
        shape = std::vector<size_t>(numpy_array.shape(), numpy_array.shape() + numpy_array.ndim());
    }
    // Handle arrays using the __cuda_array_interface__ (e.g. cupy or pytorch gpu tensors)
    else if (py::hasattr(array, "__cuda_array_interface__"))
    {
        auto cuda_interface = array.attr("__cuda_array_interface__");
        raw_ptr = reinterpret_cast<T *>(cuda_interface["data"].cast<std::pair<size_t, bool>>().first);
        shape = cuda_interface["shape"].cast<std::vector<size_t>>();
    }
    // Handle arrays using the __array_interface__ (Python Array API or array_api_strict)
    else
    {
        throw std::invalid_argument("Unsupported array type. Must have __dlpack__, __cuda_array_interface__ or be numpy.");
    }

    return {raw_ptr, shape};
}

// Wrapper for joseph3d_fwd
void joseph3d_fwd_py(py::object xstart,
                     py::object xend,
                     py::object img,
                     py::object img_origin,
                     py::object voxsize,
                     py::object p,
                     int device_id = 0,
                     int threadsperblock = 64)
{
    // Extract raw pointers and shapes
    auto [xstart_ptr, xstart_shape] = extract_pointer_and_shape<float>(xstart);
    auto [xend_ptr, xend_shape] = extract_pointer_and_shape<float>(xend);
    auto [img_ptr, img_shape] = extract_pointer_and_shape<float>(img);
    auto [img_origin_ptr, img_origin_shape] = extract_pointer_and_shape<float>(img_origin);
    auto [voxsize_ptr, voxsize_shape] = extract_pointer_and_shape<float>(voxsize);
    auto [p_ptr, p_shape] = extract_pointer_and_shape<float>(p);

    // Validate shapes
    if (xstart_shape.size() < 2 || xstart_shape[1] != 3)
    {
        throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");
    }
    if (xend_shape.size() < 2 || xend_shape[1] != 3)
    {
        throw std::invalid_argument("xend must have at least 2 dims and shape (..., 3)");
    }
    if (img_shape.size() != 3)
    {
        throw std::invalid_argument("img must be a 3D array");
    }
    // Validate that p.shape == xstart.shape[:-1]
    if (p_shape.size() != xstart_shape.size() - 1 ||
        !std::equal(p_shape.begin(), p_shape.end(), xstart_shape.begin()))
    {
        throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
    }
    if (img_origin_shape.size() != 1 || img_origin_shape[0] != 3)
    {
        throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
    }
    if (voxsize_shape.size() != 1 || voxsize_shape[0] != 3)
    {
        throw std::invalid_argument("voxsize must be a 1D array with 3 elements");
    }

    // Calculate nlors using xstart_shape (multiply shape except the last dimension)
    size_t nlors = std::accumulate(xstart_shape.begin(), xstart_shape.end() - 1, 1, std::multiplies<size_t>());
    int img_dim[3] = {static_cast<int>(img_shape[0]), static_cast<int>(img_shape[1]), static_cast<int>(img_shape[2])};
    size_t nvoxels = img_dim[0] * img_dim[1] * img_dim[2];

    // Call the C++ function
    joseph3d_fwd(xstart_ptr, xend_ptr, img_ptr, img_origin_ptr, voxsize_ptr, p_ptr, nvoxels, nlors, img_dim, device_id, threadsperblock);
}

// Wrapper for joseph3d_back
void joseph3d_back_py(py::object xstart,
                      py::object xend,
                      py::object img,
                      py::object img_origin,
                      py::object voxsize,
                      py::object p,
                      int device_id = 0,
                      int threadsperblock = 64)
{
    // Extract raw pointers and shapes
    auto [xstart_ptr, xstart_shape] = extract_pointer_and_shape<float>(xstart);
    auto [xend_ptr, xend_shape] = extract_pointer_and_shape<float>(xend);
    auto [img_ptr, img_shape] = extract_pointer_and_shape<float>(img);
    auto [img_origin_ptr, img_origin_shape] = extract_pointer_and_shape<float>(img_origin);
    auto [voxsize_ptr, voxsize_shape] = extract_pointer_and_shape<float>(voxsize);
    auto [p_ptr, p_shape] = extract_pointer_and_shape<float>(p);

    // Validate shapes
    if (xstart_shape.size() < 2 || xstart_shape[1] != 3)
    {
        throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");
    }
    if (xend_shape.size() < 2 || xend_shape[1] != 3)
    {
        throw std::invalid_argument("xend must have at least 2 dims and shape (..., 3)");
    }
    if (img_shape.size() != 3)
    {
        throw std::invalid_argument("img must be a 3D array");
    }
    // Validate that p.shape == xstart.shape[:-1]
    if (p_shape.size() != xstart_shape.size() - 1 ||
        !std::equal(p_shape.begin(), p_shape.end(), xstart_shape.begin()))
    {
        throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
    }
    if (img_origin_shape.size() != 1 || img_origin_shape[0] != 3)
    {
        throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
    }
    if (voxsize_shape.size() != 1 || voxsize_shape[0] != 3)
    {
        throw std::invalid_argument("voxsize must be a 1D array with 3 elements");
    }

    // Calculate nlors using xstart_shape (multiply shape except the last dimension)
    size_t nlors = std::accumulate(xstart_shape.begin(), xstart_shape.end() - 1, 1, std::multiplies<size_t>());

    int img_dim[3] = {static_cast<int>(img_shape[0]), static_cast<int>(img_shape[1]), static_cast<int>(img_shape[2])};
    size_t nvoxels = img_dim[0] * img_dim[1] * img_dim[2];

    // Call the C++ function
    joseph3d_back(xstart_ptr, xend_ptr, img_ptr, img_origin_ptr, voxsize_ptr, p_ptr, nvoxels, nlors, img_dim, device_id, threadsperblock);
}

// Pybind11 module definition
PYBIND11_MODULE(parallelproj_pybind, m)
{
    m.doc() = "Python bindings for parallelproj";

    m.def("joseph3d_fwd", &joseph3d_fwd_py, "Forward projection",
          py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
          py::arg("voxsize"), py::arg("p"), py::arg("device_id") = 0, py::arg("threadsperblock") = 64);

    m.def("joseph3d_back", &joseph3d_back_py, "Back projection",
          py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
          py::arg("voxsize"), py::arg("p"), py::arg("device_id") = 0, py::arg("threadsperblock") = 64);
}
