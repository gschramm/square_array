import ctypes
from array_api_compat import array_namespace, device, size

def get_data_ptr(arr):
    """
    Returns the raw data pointer from a NumPy, CuPy, or PyTorch array.

    Parameters:
        arr : array object (NumPy, CuPy, or Torch tensor)

    Returns:
        int : pointer to underlying data buffer
    """

    ptr = None

    if hasattr(arr, "_array"):  # numpy.array_api object
        ptr = arr._array.ctypes.data
    elif hasattr(arr, "data_ptr"): # pytorch
        ptr =  arr.data_ptr()
    elif hasattr(arr, "data") and hasattr(arr.data, "ptr"): # cupy
        ptr = arr.data.ptr
    elif hasattr(arr, "__array_interface__"): # numpy
        ptr = arr.__array_interface__["data"][0]
    else:
        raise TypeError("Unsupported array type or missing data pointer.")

    return ptr

# Load shared library
lib = ctypes.CDLL("./build/libsquare_array.so")

# Define signature for: void square_array(float*, size_t, float*)
lib.square_array.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p
]
lib.square_array.restype = None

def square_array_backend(arr):
    """
    Calls the C/CUDA backend on an Array API-compatible array.
    Parameters:
        arr : Array API-compatible object (NumPy, CuPy, Torch, etc.)
    Returns:
        float : sum of squares
    """
    xp = array_namespace(arr)
    dev = device(arr)

    # Allocate sum array on the same device
    sum_arr = xp.zeros(1, dtype=xp.float32, device=dev)

    # Get raw pointers
    ptr = get_data_ptr(arr)
    sum_ptr = get_data_ptr(sum_arr)

    # Call C/CUDA function
    lib.square_array(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size(arr)),
        ctypes.c_void_p(sum_ptr)
    )

    return float(sum_arr[0])

