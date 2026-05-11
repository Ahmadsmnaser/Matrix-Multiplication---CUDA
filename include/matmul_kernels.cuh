#pragma once

#include <cuda_runtime.h>

#define TILE_SIZE 32

enum KernelType
{
    NAIVE,
    TILED
};

bool launch_matmul_kernel(KernelType kernelType,
                          dim3 gridSize,
                          dim3 blockSize,
                          const float *d_A,
                          const float *d_B,
                          float *d_C,
                          int N);

