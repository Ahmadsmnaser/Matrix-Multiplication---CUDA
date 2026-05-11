#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

enum KernelType
{
    NAIVE,
    TILED,
    CUBLAS
};

__global__ void naive_matmul_kernel(const float *A, const float *B, float *C, int N)
{
    // Each thread computes one element C[row][col].
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check for matrix sizes not divisible by block size.
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
    // Shared memory tiles for A and B.
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

        // Load one tile from A into shared memory.
        if (row < N && tiledColA < N)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tiledColA];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load one tile from B into shared memory.
        if (tiledRowB < N && col < N)
        {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile.
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

void cpu_matmul(const std::vector<float> &A,
                const std::vector<float> &B,
                std::vector<float> &C,
                int N)
{
    // CPU reference implementation for correctness checking.
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float value = 0.0f;

            for (int k = 0; k < N; k++)
            {
                value += A[row * N + k] * B[k * N + col];
            }

            C[row * N + col] = value;
        }
    }
}

void print_matrix(const std::vector<float> &M, int N, const char *name)
{
    std::cout << name << ":\n";

    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            std::cout << M[row * N + col] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
}

bool verify_result(const std::vector<float> &cpu,
                   const std::vector<float> &gpu,
                   float tolerance = 1e-3f)
{
    // Compare CPU and GPU results with tolerance because floats are not exact.
    for (size_t i = 0; i < cpu.size(); i++)
    {
        if (std::fabs(cpu[i] - gpu[i]) > tolerance)
        {
            std::cerr << "Mismatch at index " << i
                      << ": CPU=" << cpu[i]
                      << ", GPU=" << gpu[i] << std::endl;
            return false;
        }
    }

    return true;
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

struct BenchmarkResult
{
    const char *implementation;
    int N;
    double cpuMs;
    double cpuGflops;
    double gpuMs;
    double gpuGflops;
    double speedup;
    bool verified;
};

bool run_benchmark(const char *implementation,
                   KernelType kernelType,
                   int N,
                   int iterations,
                   const dim3 &blockSize,
                   BenchmarkResult &result)
{
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_cpu(N * N, 0.0f);
    std::vector<float> C_gpu(N * N, 0.0f);

    // Deterministic initialization.
    // With A=1 and B=1, every output element should equal N.
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, C_cpu, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuElapsed = cpuStop - cpuStart;
    double cpuMs = cpuElapsed.count();

    double ops = 2.0 * static_cast<double>(N) * N * N;
    double cpuSec = cpuMs / 1000.0;
    double cpuGflops = ops / cpuSec / 1e9;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    cudaError_t err;

    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for A: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for B: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return false;
    }

    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for C: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return false;
    }

    err = cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy A to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy B to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaMemset(d_C, 0, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to initialize C on device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Warm-up launch.
    if (!launch_matmul_kernel(kernelType, gridSize, blockSize, d_A, d_B, d_C, N))
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Warm-up kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Warm-up kernel execution failed: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    cudaEvent_t start, stop;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to create start event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to create stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventRecord(start);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to record start event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    for (int i = 0; i < iterations; i++)
    {
        if (!launch_matmul_kernel(kernelType, gridSize, blockSize, d_A, d_B, d_C, N))
        {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return false;
        }
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Benchmark kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventRecord(stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to record stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to synchronize stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    float totalMs = 0.0f;

    err = cudaEventElapsedTime(&totalMs, start, stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get elapsed time: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    float avgMs = totalMs / iterations;
    double gpuSec = avgMs / 1000.0;
    double gpuGflops = ops / gpuSec / 1e9;
    double speedup = cpuMs / avgMs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy C from device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    if (N <= 16)
    {
        print_matrix(A, N, "Matrix A");
        print_matrix(B, N, "Matrix B");
        print_matrix(C_cpu, N, "Matrix C_cpu");
        print_matrix(C_gpu, N, "Matrix C_gpu");
    }

    bool verified = verify_result(C_cpu, C_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    result.implementation = implementation;
    result.N = N;
    result.cpuMs = cpuMs;
    result.cpuGflops = cpuGflops;
    result.gpuMs = avgMs;
    result.gpuGflops = gpuGflops;
    result.speedup = speedup;
    result.verified = verified;

    return true;
}

void print_result(const BenchmarkResult &result)
{
    std::cout << std::left
              << std::setw(16) << result.implementation
              << std::setw(8) << result.N
              << std::setw(14) << std::fixed << std::setprecision(3) << result.cpuMs
              << std::setw(16) << std::fixed << std::setprecision(3) << result.cpuGflops
              << std::setw(12) << std::fixed << std::setprecision(3) << result.gpuMs
              << std::setw(16) << std::fixed << std::setprecision(3) << result.gpuGflops
              << std::setw(12) << std::fixed << std::setprecision(3) << result.speedup
              << std::setw(10) << (result.verified ? "YES" : "NO")
              << "\n";
}

bool run_cublas_benchmark(int N, int iterations, BenchmarkResult &result){
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_cpu(N * N, 0.0f);
    std::vector<float> C_gpu(N * N, 0.0f);

    // Keep this initialization identical to the custom kernel benchmarks.
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, C_cpu, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuElapsed = cpuStop - cpuStart;
    double cpuMs = cpuElapsed.count();

    double ops = 2.0 * static_cast<double>(N) * N * N;
    double cpuSec = cpuMs / 1000.0;
    double cpuGflops = ops / cpuSec / 1e9;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    cudaError_t err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for cuBLAS A: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for cuBLAS B: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return false;
    }

    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for cuBLAS C: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return false;
    }

    err = cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy cuBLAS A to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy cuBLAS B to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create cuBLAS handle: " << status << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS expects column-major matrices. Swapping A and B computes the
    // correct result for our row-major memory layout: C = A * B.
    status = cublasSgemm(handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         N,
                         N,
                         N,
                         &alpha,
                         d_B,
                         N,
                         d_A,
                         N,
                         &beta,
                         d_C,
                         N);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS warm-up failed: " << status << std::endl;
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "cuBLAS warm-up execution failed: "
                  << cudaGetErrorString(err) << std::endl;
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to create cuBLAS start event: "
                  << cudaGetErrorString(err) << std::endl;
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to create cuBLAS stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventRecord(start);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to record cuBLAS start event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    for (int i = 0; i < iterations; i++)
    {
        status = cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N,
                             N,
                             N,
                             &alpha,
                             d_B,
                             N,
                             d_A,
                             N,
                             &beta,
                             d_C,
                             N);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cuBLAS benchmark call failed: " << status << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cublasDestroy(handle);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return false;
        }
    }

    err = cudaEventRecord(stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to record cuBLAS stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to synchronize cuBLAS stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    float totalMs = 0.0f;
    err = cudaEventElapsedTime(&totalMs, start, stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get cuBLAS elapsed time: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    float avgMs = totalMs / iterations;
    double gpuSec = avgMs / 1000.0;
    double gpuGflops = ops / gpuSec / 1e9;
    double speedup = cpuMs / avgMs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy cuBLAS C from device: "
                  << cudaGetErrorString(err) << std::endl;
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    bool verified = verify_result(C_cpu, C_gpu);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    result.implementation = "cuBLAS";
    result.N = N;
    result.cpuMs = cpuMs;
    result.cpuGflops = cpuGflops;
    result.gpuMs = avgMs;
    result.gpuGflops = gpuGflops;
    result.speedup = speedup;
    result.verified = verified;

    return true;
}

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "CUDA Device Count: " << deviceCount << "\n";

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable GPU found.\n";
        return -1;
    }

    const int sizes[] = {256, 512, 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    const int iterations = 10;
    const dim3 blockSize(TILE_SIZE, TILE_SIZE);

    std::cout << "=== Matrix Multiplication Benchmark ===\n";
    std::cout << "Tile/Block: " << blockSize.x << " x " << blockSize.y << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    std::cout << std::left
              << std::setw(16) << "Implementation"
              << std::setw(8) << "N"
              << std::setw(14) << "CPU ms"
              << std::setw(16) << "CPU GFLOPS"
              << std::setw(12) << "GPU ms"
              << std::setw(16) << "GPU GFLOPS"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Verified"
              << "\n";

    for (int i = 0; i < numSizes; i++)
    {
        BenchmarkResult naiveResult{};
        if (!run_benchmark("Naive", NAIVE, sizes[i], iterations, blockSize, naiveResult))
        {
            return -1;
        }
        print_result(naiveResult);

        BenchmarkResult tiledResult{};
        if (!run_benchmark("Tiled", TILED, sizes[i], iterations, blockSize, tiledResult))
        {
            return -1;
        }
        print_result(tiledResult);

        BenchmarkResult cublasResult{};
        if (!run_cublas_benchmark(sizes[i], iterations, cublasResult))
        {
            return -1;
        }
        print_result(cublasResult);
    }

    return 0;
}
