#include "parallelproj.h"
#include "debug.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__device__ unsigned char ray_cube_intersection_cuda(float orig0,
                                                    float orig1,
                                                    float orig2,
                                                    float bounds0_min,
                                                    float bounds1_min,
                                                    float bounds2_min,
                                                    float bounds0_max,
                                                    float bounds1_max,
                                                    float bounds2_max,
                                                    float rdir0,
                                                    float rdir1,
                                                    float rdir2,
                                                    float* t1,
                                                    float* t2){
  // the inverse of the directional vector
  // using the inverse of the directional vector and IEEE floating point arith standard 754
  // makes sure that 0's in the directional vector are handled correctly 
  float invdir0 = 1.f/rdir0;
  float invdir1 = 1.f/rdir1;
  float invdir2 = 1.f/rdir2;
  
  unsigned char intersec = 1;
  
  float t11, t12, t21, t22; 

  if (invdir0 >= 0){
    *t1  = (bounds0_min - orig0) * invdir0;
    *t2  = (bounds0_max - orig0) * invdir0; 
  }
  else{
    *t1  = (bounds0_max - orig0) * invdir0;
    *t2  = (bounds0_min - orig0) * invdir0;
  }
  
  if (invdir1 >= 0){
    t11 = (bounds1_min - orig1) * invdir1; 
    t12 = (bounds1_max - orig1) * invdir1; 
  }
  else{
    t11 = (bounds1_max - orig1) * invdir1;
    t12 = (bounds1_min - orig1) * invdir1; 
  }
  
  if ((*t1 > t12) || (t11 > *t2)){intersec = 0;}
  if (t11 > *t1){*t1 = t11;}
  if (t12 < *t2){*t2 = t12;}
  
  if (invdir2 >= 0){
    t21 = (bounds2_min - orig2) * invdir2; 
    t22 = (bounds2_max - orig2) * invdir2;
  } 
  else{
    t21 = (bounds2_max - orig2) * invdir2; 
    t22 = (bounds2_min - orig2) * invdir2;
  } 
  
  if ((*t1 > t22) || (t21 > *t2)){intersec = 0;}
  if (t21 > *t1){*t1 = t21;}
  if (t22 < *t2){*t2 = t22;} 

  return(intersec);
}


__global__ void joseph3d_fwd_kernel(const float *xstart, 
                                    const float *xend, 
                                    const float *img,
                                    const float *img_origin, 
                                    const float *voxsize, 
                                    float *p,
                                    long long nlors, 
                                    const int *img_dim)
{
  long long i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < nlors)
  {
    int n0 = img_dim[0];
    int n1 = img_dim[1];
    int n2 = img_dim[2];

    float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    int i0, i1, i2;
    int i0_floor, i1_floor, i2_floor;
    int i0_ceil, i1_ceil, i2_ceil;
    float x_pr0, x_pr1, x_pr2;
    float tmp_0, tmp_1, tmp_2;

    float toAdd, cf;

    float xstart0 = xstart[i*3 + 0];
    float xstart1 = xstart[i*3 + 1];
    float xstart2 = xstart[i*3 + 2];

    float xend0 = xend[i*3 + 0];
    float xend1 = xend[i*3 + 1];
    float xend2 = xend[i*3 + 2];

    float voxsize0 = voxsize[0];
    float voxsize1 = voxsize[1];
    float voxsize2 = voxsize[2];

    float img_origin0 = img_origin[0];
    float img_origin1 = img_origin[1];
    float img_origin2 = img_origin[2];

    unsigned char intersec;
    float t1, t2;
    float istart_f, iend_f, tmp;
    int   istart, iend;

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend0 - xstart0;
    d1 = xend1 - xstart1;
    d2 = xend2 - xstart2;

    //-----------
    //--- test whether ray and cube intersect
    intersec = ray_cube_intersection_cuda(xstart0, xstart1, xstart2, 
                                          img_origin0 - 1*voxsize0, img_origin1 - 1*voxsize1, img_origin2 - 1*voxsize2,
                                          img_origin0 + n0*voxsize0, img_origin1 + n1*voxsize1, img_origin2 + n2*voxsize2,
                                          d0, d1, d2, &t1, &t2);

    if (intersec == 1)
    {
      d0_sq = d0*d0;
      d1_sq = d1*d1;
      d2_sq = d2*d2;

      lsq = d0_sq + d1_sq + d2_sq;

      cos0_sq = d0_sq / lsq;
      cos1_sq = d1_sq / lsq;
      cos2_sq = d2_sq / lsq;

      direction = 0;
      if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
      {
        direction = 1;
      }
      else
      {
        if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
        {
          direction = 2;
        }
      }
 
      if (direction == 0)
      {
        cf = voxsize0 / sqrtf(cos0_sq);

        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart0 + t1*d0 - img_origin0) / voxsize0;
        iend_f   = (xstart0 + t2*d0 - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        if (istart < 0){istart = 0;}
        if (iend >= n0){iend = n0;}

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart0 - img_origin0) / voxsize0;
        iend_f   = (xend0   - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f)){istart = (int)floor(istart_f);}
        if (iend >= (int)ceil(iend_f)){iend = (int)ceil(iend_f);}
        //---

        for(i0 = istart; i0 < iend; i0++)
        {
          // get the indices where the ray intersects the image plane
          x_pr1 = xstart1 + (img_origin0 + i0*voxsize0 - xstart0)*d1 / d0;
          x_pr2 = xstart2 + (img_origin0 + i0*voxsize0 - xstart0)*d2 / d0;
  
          i1_floor = (int)floor((x_pr1 - img_origin1)/voxsize1);
          i1_ceil  = i1_floor + 1;
  
          i2_floor = (int)floor((x_pr2 - img_origin2)/voxsize2);
          i2_ceil  = i2_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;
          tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;

          toAdd = 0;

          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_floor + i2_floor] * (1 - tmp_1) * (1 - tmp_2);
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_ceil + i2_floor] * tmp_1 * (1 - tmp_2);
          }
          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_floor + i2_ceil] * (1 - tmp_1) * tmp_2;
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_ceil + i2_ceil] * tmp_1 * tmp_2;
          }

          if(toAdd != 0){p[i] += (cf * toAdd);}
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 1)
      {
        cf = voxsize1 / sqrtf(cos1_sq);

        // case where ray is most parallel to the 1 axis
        // we step through the volume along the 1 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart1 + t1*d1 - img_origin1) / voxsize1;
        iend_f   = (xstart1 + t2*d1 - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        if (istart < 0){istart = 0;}
        if (iend >= n1){iend = n1;}

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart1 - img_origin1) / voxsize1;
        iend_f   = (xend1   - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f)){istart = (int)floor(istart_f);}
        if (iend >= (int)ceil(iend_f)){iend = (int)ceil(iend_f);}
        //---

        for (i1 = istart; i1 < iend; i1++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin1 + i1*voxsize1 - xstart1)*d0 / d1;
          x_pr2 = xstart2 + (img_origin1 + i1*voxsize1 - xstart1)*d2 / d1;
  
          i0_floor = (int)floor((x_pr0 - img_origin0)/voxsize0);
          i0_ceil  = i0_floor + 1; 
  
          i2_floor = (int)floor((x_pr2 - img_origin2)/voxsize2);
          i2_ceil  = i2_floor + 1;
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
          tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;

          toAdd = 0;

          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1 + i2_floor] * (1 - tmp_0) * (1 - tmp_2);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1 + i2_floor] * tmp_0 * (1 - tmp_2);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1 + i2_ceil] * (1 - tmp_0) * tmp_2;
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1 + i2_ceil] * tmp_0 * tmp_2;
          }

          if(toAdd != 0){p[i] += (cf * toAdd);}
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 2)
      {
        cf = voxsize2 / sqrtf(cos2_sq);

        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction

        //--- check where ray enters / leaves cube
        istart_f = (xstart2 + t1*d2 - img_origin2) / voxsize2;
        iend_f   = (xstart2 + t2*d2 - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        if (istart < 0){istart = 0;}
        if (iend >= n2){iend = n2;}

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart2 - img_origin2) / voxsize2;
        iend_f   = (xend2   - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f)){istart = (int)floor(istart_f);}
        if (iend >= (int)ceil(iend_f)){iend = (int)ceil(iend_f);}
        //---

        for(i2 = istart; i2 < iend; i2++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin2 + i2*voxsize2 - xstart2)*d0 / d2;
          x_pr1 = xstart1 + (img_origin2 + i2*voxsize2 - xstart2)*d1 / d2;
  
          i0_floor = (int)floor((x_pr0 - img_origin0)/voxsize0);
          i0_ceil  = i0_floor + 1;
  
          i1_floor = (int)floor((x_pr1 - img_origin1)/voxsize1);
          i1_ceil  = i1_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
          tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;

          toAdd = 0;

          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1_floor + i2] * (1 - tmp_0) * (1 - tmp_1);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1_floor + i2] * tmp_0 * (1 - tmp_1);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) & (i1_ceil < n1))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1_ceil + i2] * (1 - tmp_0) * tmp_1;
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1_ceil + i2] * tmp_0 * tmp_1;
          }

          if(toAdd != 0){p[i] += (cf * toAdd);}
        }
      }
    }
  }
}


extern "C" 
void joseph3d_fwd(const float *xstart, 
                  const float *xend, 
                  const float *img,
                  const float *img_origin, 
                  const float *voxsize, 
                  float *p,
                  long long nlors, 
                  const int *img_dim,
                  int device_id,
                  int threadsperblock)
{

    const float* d_xstart = nullptr;
    const float* d_xend = nullptr;
    const float* d_img = nullptr;
    const float* d_img_origin = nullptr;
    const float* d_voxsize = nullptr;
    float* d_p = nullptr;
    const int* d_img_dim = nullptr;

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

    if (err == cudaSuccess && (xstart_attr.type == cudaMemoryTypeManaged)){
        is_cuda_managed_ptr = true;
        DEBUG_PRINT("Managed array is on device : %d\n", xstart_attr.device);
    }
    // else throw error
    else{
        needs_copy_back = true;
        throw std::runtime_error("Unsupported pointer type");
    }

    if (is_cuda_managed_ptr){
    // all arrays are cuda malloc managed, so no need to copy to the device
        d_xstart = xstart;
        d_xend = xend;
        d_img = img;
        d_img_origin = img_origin;
        d_voxsize = voxsize;
        d_p = p;
        d_img_dim = img_dim;
    } else {
        DEBUG_PRINT("COPYING HOST TO DEVICE");
    }

    // get and print the current cuda device ID
    int current_device_id;
    cudaGetDevice(&current_device_id);
    DEBUG_PRINT("Using CUDA device: %d\n", current_device_id);


    int num_blocks = (int)((nlors + threadsperblock- 1) / threadsperblock);
    joseph3d_fwd_kernel<<<num_blocks,threadsperblock>>>(d_xstart, d_xend, d_img, 
                                         d_img_origin, d_voxsize, 
                                         d_p, nlors, d_img_dim);
    cudaDeviceSynchronize();

    //if (needs_copy_back) {
    //    cudaMemcpy(array, device_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    //    cudaFree(device_array);
    //}
}

