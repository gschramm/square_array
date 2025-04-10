#include "parallelproj.h"
#include "joseph3d_fwd_worker.h"
#include "debug.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void joseph3d_fwd_kernel(const float *xstart,
                                    const float *xend,
                                    const float *img,
                                    const float *img_origin,
                                    const float *voxsize,
                                    float *p,
                                    size_t nlors,
                                    const int *img_dim)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nlors)
    {
        joseph3d_fwd_worker(i, xstart, xend, img, img_origin, voxsize, p, img_dim);
    }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void joseph3d_fwd(const float *xstart,
                  const float *xend,
                  const float *img,
                  const float *img_origin,
                  const float *voxsize,
                  float *p,
                  size_t nlors,
                  const int *img_dim,
                  int device_id,
                  int threadsperblock)
{

    const float *d_xstart = nullptr;
    const float *d_xend = nullptr;
    const float *d_img = nullptr;
    const float *d_img_origin = nullptr;
    const float *d_voxsize = nullptr;
    float *d_p = nullptr;
    const int *d_img_dim = nullptr;

    // get pointer attributes of all input and output arrays
    cudaPointerAttributes xstart_attr;
    cudaError_t err = cudaPointerGetAttributes(&xstart_attr, xstart);
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    // TODO get attributes of all other arrays
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    bool needs_copy_back = false;
    bool is_cuda_managed_ptr = false;

    if (err == cudaSuccess && (xstart_attr.type == cudaMemoryTypeManaged))
    {
        is_cuda_managed_ptr = true;
        DEBUG_PRINT("Managed array is on device : %d\n", xstart_attr.device);
    }
    // else throw error
    else
    {
        needs_copy_back = true;
        throw std::runtime_error("Unsupported pointer type");
    }

    if (is_cuda_managed_ptr)
    {
        // all arrays are cuda malloc managed, so no need to copy to the device
        d_xstart = xstart;
        d_xend = xend;
        d_img = img;
        d_img_origin = img_origin;
        d_voxsize = voxsize;
        d_p = p;
        d_img_dim = img_dim;
    }
    else
    {
        DEBUG_PRINT("COPYING HOST TO DEVICE");
    }

    // get and print the current cuda device ID
    int current_device_id;
    cudaGetDevice(&current_device_id);
    DEBUG_PRINT("Using CUDA device: %d\n", current_device_id);

    int num_blocks = (int)((nlors + threadsperblock - 1) / threadsperblock);
    joseph3d_fwd_kernel<<<num_blocks, threadsperblock>>>(d_xstart, d_xend, d_img,
                                                         d_img_origin, d_voxsize,
                                                         d_p, nlors, d_img_dim);
    cudaDeviceSynchronize();

    // if (needs_copy_back) {
    //     cudaMemcpy(array, device_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaFree(device_array);
    // }
}
