#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/// Wraps every CUDA API call to check for errors.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/// Checks for kernel launch errors (placed after kernel<<<>>> calls).
#define CUDA_KERNEL_CHECK()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace neurograph::gpu {

/// Standard block size for 1D kernels.
constexpr int BLOCK_SIZE = 256;

/// Computes the number of blocks needed to cover n elements.
inline int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

/// Prints basic GPU device information.
inline void print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return;
    }
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s (SM %d.%d, %zu MB, %d SMs)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024 * 1024),
               prop.multiProcessorCount);
    }
}

} // namespace neurograph::gpu
