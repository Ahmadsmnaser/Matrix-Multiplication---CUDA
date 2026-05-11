#include "matmul_kernels.cuh"

#include <iostream>

__global__ void naive_matmul_kernel(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float value = 0.0f;

        for (int k = 0; k < N; k++)
        {
            value += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = value;
    }
}

__global__ void tiled_matmul_kernel(const float *A, const float *B, float *C, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        int tiledColA = t * TILE_SIZE + threadIdx.x;
        int tiledRowB = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < N && tiledColA < N) ? A[row * N + tiledColA] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (tiledRowB < N && col < N) ? B[tiledRowB * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = value;
    }
}

bool launch_matmul_kernel(KernelType kernelType,
                          dim3 gridSize,
                          dim3 blockSize,
                          const float *d_A,
                          const float *d_B,
                          float *d_C,
                          int N)
{
    if (kernelType == NAIVE)
    {
        naive_matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        return true;
    }

    if (kernelType == TILED)
    {
        tiled_matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        return true;
    }

    std::cerr << "Unknown kernel type." << std::endl;
    return false;
}

