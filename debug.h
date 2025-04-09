// debug.h
#pragma once

#include <cstdio>

// Safe printf wrapper
#define DPRINTF(...)         \
    do                       \
    {                        \
        printf(__VA_ARGS__); \
    } while (0)

// Conditional debug print
#ifdef DEBUG
#define DEBUG_PRINT(...) DPRINTF(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) \
    do                   \
    {                    \
    } while (0)
#endif
