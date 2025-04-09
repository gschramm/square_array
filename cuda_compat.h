#pragma once

#ifdef __CUDACC__
#define WORKER_QUALIFIER __device__
#else
#define WORKER_QUALIFIER
#endif
