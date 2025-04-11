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

    // handle xstart (read only)
    float *d_xstart = nullptr;
    cudaPointerAttributes xstart_attr;
    cudaError_t err_xstart = cudaPointerGetAttributes(&xstart_attr, xstart);
    bool free_xstart = false;
    if (err_xstart == cudaSuccess && (xstart_attr.type == cudaMemoryTypeManaged))
    {
        cudaMemPrefetchAsync(xstart, sizeof(float) * nlors * 3, device_id);
        cudaMemAdvise(xstart, sizeof(float) * nlors * 3, cudaMemAdviseSetReadMostly, device_id);
    }

    if (err_xstart == cudaSuccess && (xstart_attr.type == cudaMemoryTypeManaged ||
                                      xstart_attr.type == cudaMemoryTypeDevice))
    {
        d_xstart = const_cast<float *>(xstart);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_xstart, sizeof(float) * nlors * 3);
        cudaMemcpy(d_xstart, xstart, sizeof(float) * nlors * 3, cudaMemcpyHostToDevice);
        free_xstart = true;
    }

    // handle xend (read only)
    float *d_xend = nullptr;
    cudaPointerAttributes xend_attr;
    cudaError_t err_xend = cudaPointerGetAttributes(&xend_attr, xend);
    bool free_xend = false;
    if (err_xend == cudaSuccess && xend_attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(xend, sizeof(float) * nlors * 3, device_id);
        cudaMemAdvise(xend, sizeof(float) * nlors * 3, cudaMemAdviseSetReadMostly, device_id);
    }

    if (err_xend == cudaSuccess && (xend_attr.type == cudaMemoryTypeManaged ||
                                    xend_attr.type == cudaMemoryTypeDevice))
    {
        d_xend = const_cast<float *>(xend);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_xend, sizeof(float) * nlors * 3);
        cudaMemcpy(d_xend, xend, sizeof(float) * nlors * 3, cudaMemcpyHostToDevice);
        free_xend = true;
    }

    // handle img (read only)
    float *d_img = nullptr;
    cudaPointerAttributes img_attr;
    cudaError_t err_img = cudaPointerGetAttributes(&img_attr, img);
    bool free_img = false;
    if (err_img == cudaSuccess && img_attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(img, sizeof(float) * img_dim[0] * img_dim[1] * img_dim[2], device_id);
        cudaMemAdvise(img, sizeof(float) * img_dim[0] * img_dim[1] * img_dim[2], cudaMemAdviseSetReadMostly, device_id);
    }

    if (err_img == cudaSuccess && (img_attr.type == cudaMemoryTypeManaged ||
                                   img_attr.type == cudaMemoryTypeDevice))
    {
        d_img = const_cast<float *>(img);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_img, sizeof(float) * img_dim[0] * img_dim[1] * img_dim[2]);
        cudaMemcpy(d_img, img, sizeof(float) * img_dim[0] * img_dim[1] * img_dim[2], cudaMemcpyHostToDevice);
        free_img = true;
    }

    // handle img_origin (read only)
    float *d_img_origin = nullptr;
    cudaPointerAttributes img_origin_attr;
    cudaError_t err_img_origin = cudaPointerGetAttributes(&img_origin_attr, img_origin);
    bool free_img_origin = false;
    if (err_img_origin == cudaSuccess && img_origin_attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(img_origin, sizeof(float) * 3, device_id);
        cudaMemAdvise(img_origin, sizeof(float) * 3, cudaMemAdviseSetReadMostly, device_id);
    }

    if (err_img_origin == cudaSuccess && (img_origin_attr.type == cudaMemoryTypeManaged ||
                                          img_origin_attr.type == cudaMemoryTypeDevice))
    {
        d_img_origin = const_cast<float *>(img_origin);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_img_origin, sizeof(float) * 3);
        cudaMemcpy(d_img_origin, img_origin, sizeof(float) * 3, cudaMemcpyHostToDevice);
        free_img_origin = true;
    }

    // handle voxsize (read only)
    float *d_voxsize = nullptr;
    cudaPointerAttributes voxsize_attr;
    cudaError_t err_voxsize = cudaPointerGetAttributes(&voxsize_attr, voxsize);
    bool free_voxsize = false;
    if (err_voxsize == cudaSuccess && voxsize_attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(voxsize, sizeof(float) * 3, device_id);
        cudaMemAdvise(voxsize, sizeof(float) * 3, cudaMemAdviseSetReadMostly, device_id);
    }

    if (err_voxsize == cudaSuccess && (voxsize_attr.type == cudaMemoryTypeManaged ||
                                       voxsize_attr.type == cudaMemoryTypeDevice))
    {
        d_voxsize = const_cast<float *>(voxsize);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_voxsize, sizeof(float) * 3);
        cudaMemcpy(d_voxsize, voxsize, sizeof(float) * 3, cudaMemcpyHostToDevice);
        free_voxsize = true;
    }

    // handle p (write)
    float *d_p = nullptr;
    cudaPointerAttributes p_attr;
    cudaError_t err_p = cudaPointerGetAttributes(&p_attr, p);
    bool free_p = false;
    if (err_p == cudaSuccess && p_attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(p, sizeof(float) * nlors, device_id);
        cudaMemAdvise(p, sizeof(float) * nlors, cudaMemAdviseSetAccessedBy, device_id);
    }

    if (err_p == cudaSuccess && (p_attr.type == cudaMemoryTypeManaged ||
                                 p_attr.type == cudaMemoryTypeDevice))
    {
        d_p = const_cast<float *>(p);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_p, sizeof(float) * nlors);
        cudaMemcpy(d_p, p, sizeof(float) * nlors, cudaMemcpyHostToDevice);
        free_p = true;
    }

    // handle img_dim (read only)
    int *d_img_dim = nullptr;
    cudaPointerAttributes img_dim_attr;
    cudaError_t err_img_dim = cudaPointerGetAttributes(&img_dim_attr, img_dim);
    bool free_img_dim = false;
    if (err_img_dim == cudaSuccess && img_dim_attr.type == cudaMemoryTypeManaged)
    {
        cudaMemPrefetchAsync(img_dim, sizeof(int) * 3, device_id);
        cudaMemAdvise(img_dim, sizeof(int) * 3, cudaMemAdviseSetReadMostly, device_id);
    }

    if (err_img_dim == cudaSuccess && (img_dim_attr.type == cudaMemoryTypeManaged ||
                                       img_dim_attr.type == cudaMemoryTypeDevice))
    {
        d_img_dim = const_cast<int *>(img_dim);
    }
    else
    {
        // host pointer case, transfer to device
        cudaMalloc(&d_img_dim, sizeof(int) * 3);
        cudaMemcpy(d_img_dim, img_dim, sizeof(int) * 3, cudaMemcpyHostToDevice);
        free_img_dim = true;
    }

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
