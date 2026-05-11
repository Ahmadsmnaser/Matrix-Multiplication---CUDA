#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matmul_cpu.hpp"
#include "matmul_kernels.cuh"

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

struct CpuReference
{
    std::vector<float> C;
    double ms;
    double gflops;
};

struct DeviceBuffers
{
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
};

double operation_count(int N)
{
    return 2.0 * static_cast<double>(N) * N * N;
}

double calculate_gflops(double ops, double ms)
{
    return ops / (ms / 1000.0) / 1e9;
}

void initialize_matrices(std::vector<float> &A, std::vector<float> &B)
{
    for (size_t i = 0; i < A.size(); i++)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
}

CpuReference build_cpu_reference(const std::vector<float> &A,
                                 const std::vector<float> &B,
                                 int N,
                                 double ops)
{
    CpuReference reference;
    reference.C.assign(static_cast<size_t>(N) * N, 0.0f);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, reference.C, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuElapsed = cpuStop - cpuStart;
    reference.ms = cpuElapsed.count();
    reference.gflops = calculate_gflops(ops, reference.ms);

    return reference;
}

void fill_result(BenchmarkResult &result,
                 const char *implementation,
                 int N,
                 const CpuReference &reference,
                 double gpuMs,
                 double ops)
{
    result.implementation = implementation;
    result.N = N;
    result.cpuMs = reference.ms;
    result.cpuGflops = reference.gflops;
    result.gpuMs = gpuMs;
    result.gpuGflops = calculate_gflops(ops, gpuMs);
    result.speedup = reference.ms / gpuMs;
}

void free_device_buffers(DeviceBuffers &device)
{
    cudaFree(device.A);
    cudaFree(device.B);
    cudaFree(device.C);
    device.A = nullptr;
    device.B = nullptr;
    device.C = nullptr;
}

bool prepare_device_buffers(const std::vector<float> &A,
                            const std::vector<float> &B,
                            size_t bytes,
                            DeviceBuffers &device)
{
    cudaError_t err = cudaMalloc(&device.A, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for A: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&device.B, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for B: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    err = cudaMalloc(&device.C, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for C: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    err = cudaMemcpy(device.A, A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy A to device: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    err = cudaMemcpy(device.B, B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy B to device: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    err = cudaMemset(device.C, 0, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to initialize C on device: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    return true;
}

bool run_benchmark(const char *implementation,
                   KernelType kernelType,
                   int N,
                   int iterations,
                   const dim3 &blockSize,
                   BenchmarkResult &result)
{
    std::vector<float> A(static_cast<size_t>(N) * N);
    std::vector<float> B(static_cast<size_t>(N) * N);
    std::vector<float> C_gpu(static_cast<size_t>(N) * N, 0.0f);

    initialize_matrices(A, B);

    double ops = operation_count(N);
    CpuReference reference = build_cpu_reference(A, B, N, ops);

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    DeviceBuffers device;

    if (!prepare_device_buffers(A, B, bytes, device))
    {
        return false;
    }

    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Warm-up launch.
    if (!launch_matmul_kernel(kernelType, gridSize, blockSize, device.A, device.B, device.C, N))
    {
        free_device_buffers(device);
        return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Warm-up kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Warm-up kernel execution failed: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    cudaEvent_t start, stop;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to create start event: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to create stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        free_device_buffers(device);
        return false;
    }

    err = cudaEventRecord(start);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to record start event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free_device_buffers(device);
        return false;
    }

    for (int i = 0; i < iterations; i++)
    {
        if (!launch_matmul_kernel(kernelType, gridSize, blockSize, device.A, device.B, device.C, N))
        {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            free_device_buffers(device);
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
        free_device_buffers(device);
        return false;
    }

    err = cudaEventRecord(stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to record stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free_device_buffers(device);
        return false;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to synchronize stop event: "
                  << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free_device_buffers(device);
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
        free_device_buffers(device);
        return false;
    }

    float avgMs = totalMs / iterations;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    err = cudaMemcpy(C_gpu.data(), device.C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy C from device: "
                  << cudaGetErrorString(err) << std::endl;
        free_device_buffers(device);
        return false;
    }

    bool verified = verify_result(reference.C, C_gpu);

    free_device_buffers(device);

    fill_result(result, implementation, N, reference, avgMs, ops);
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
    std::vector<float> A(static_cast<size_t>(N) * N);
    std::vector<float> B(static_cast<size_t>(N) * N);
    std::vector<float> C_gpu(static_cast<size_t>(N) * N, 0.0f);

    initialize_matrices(A, B);

    double ops = operation_count(N);
    CpuReference reference = build_cpu_reference(A, B, N, ops);

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

    bool verified = verify_result(reference.C, C_gpu);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    fill_result(result, "cuBLAS", N, reference, avgMs, ops);
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
