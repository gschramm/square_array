#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void joseph3d_fwd(const float *xstart, 
                  const float *xend, 
                  const float *img,
                  const float *img_origin, 
                  const float *voxsize, 
                  float *p,
                  long long nlors, 
                  const int *img_dim,
                  int device_id = 0,
                  int threadsperblock = 64);

#ifdef __cplusplus
}
#endif
