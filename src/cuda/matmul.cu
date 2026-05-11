#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

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

int main()
{
    const int N = 4;

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_cpu(N * N, 0.0f);
    std::vector<float> C_gpu(N * N, 0.0f);

    // Deterministic initialization.
    for (int i = 0; i < N * N; i++)
    {
        A[i] = static_cast<float>(i + 1);
        B[i] = 1.0f;
    }

    cpu_matmul(A, B, C_cpu, N);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t bytes = N * N * sizeof(float);

    cudaError_t err;

    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for A: "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for B: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return -1;
    }

    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for C: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    err = cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy A to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    err = cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy B to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    err = cudaMemset(d_C, 0, bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to initialize C on device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    naive_matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution failed: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    err = cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy C from device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    print_matrix(A, N, "Matrix A");
    print_matrix(B, N, "Matrix B");
    print_matrix(C_cpu, N, "Matrix C_cpu");
    print_matrix(C_gpu, N, "Matrix C_gpu");

    if (verify_result(C_cpu, C_gpu))
    {
        std::cout << "Verification PASSED." << std::endl;
    }
    else
    {
        std::cout << "Verification FAILED." << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}