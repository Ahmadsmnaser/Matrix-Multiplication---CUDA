#include "matmul_cpu.hpp"

void cpu_matmul(const std::vector<float> &A,
                const std::vector<float> &B,
                std::vector<float> &C,
                int N)
{
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

