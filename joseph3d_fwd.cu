#include "parallelproj.h"
#include "joseph3d_fwd_worker.h"
#include "debug.h"
#include "cuda_utils.h"
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
    // Set the CUDA device
    if (device_id >= 0)
    {
        cudaSetDevice(device_id);
    }

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    // copy arrays to device if needed
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Handle xstart (read mostly)
    float *d_xstart = nullptr;
    bool free_xstart = false;
    handle_cuda_input_array(xstart, &d_xstart, sizeof(float) * nlors * 3, free_xstart, device_id, cudaMemAdviseSetReadMostly);

    // Handle xend (read mostly)
    float *d_xend = nullptr;
    bool free_xend = false;
    handle_cuda_input_array(xend, &d_xend, sizeof(float) * nlors * 3, free_xend, device_id, cudaMemAdviseSetReadMostly);

    // Handle img (read mostly)
    float *d_img = nullptr;
    bool free_img = false;
    handle_cuda_input_array(img, &d_img, sizeof(float) * img_dim[0] * img_dim[1] * img_dim[2], free_img, device_id, cudaMemAdviseSetReadMostly);

    // Handle img_origin (read mostly)
    float *d_img_origin = nullptr;
    bool free_img_origin = false;
    handle_cuda_input_array(img_origin, &d_img_origin, sizeof(float) * 3, free_img_origin, device_id, cudaMemAdviseSetReadMostly);

    // Handle voxsize (read mostly)
    float *d_voxsize = nullptr;
    bool free_voxsize = false;
    handle_cuda_input_array(voxsize, &d_voxsize, sizeof(float) * 3, free_voxsize, device_id, cudaMemAdviseSetReadMostly);

    // Handle p (write access)
    float *d_p = nullptr;
    bool free_p = false;
    handle_cuda_input_array(p, &d_p, sizeof(float) * nlors, free_p, device_id, cudaMemAdviseSetAccessedBy);

    // Handle img_dim (read mostly)
    int *d_img_dim = nullptr;
    bool free_img_dim = false;
    handle_cuda_input_array(img_dim, &d_img_dim, sizeof(int) * 3, free_img_dim, device_id, cudaMemAdviseSetReadMostly);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // launch the kernel
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG
    // get and print the current cuda device ID
    int current_device_id;
    cudaGetDevice(&current_device_id);
    DEBUG_PRINT("Using CUDA device: %d\n", current_device_id);
#endif

    int num_blocks = (int)((nlors + threadsperblock - 1) / threadsperblock);
    joseph3d_fwd_kernel<<<num_blocks, threadsperblock>>>(d_xstart, d_xend, d_img,
                                                         d_img_origin, d_voxsize,
                                                         d_p, nlors, d_img_dim);
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // free device memory if needed
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // Free device memory if it was allocated
    if (free_xstart)
        cudaFree(d_xstart);
    if (free_xend)
        cudaFree(d_xend);
    if (free_img)
        cudaFree(d_img);
    if (free_img_origin)
        cudaFree(d_img_origin);
    if (free_voxsize)
        cudaFree(d_voxsize);
    if (free_p)
        cudaFree(d_p);
    if (free_img_dim)
        cudaFree(d_img_dim);
}
