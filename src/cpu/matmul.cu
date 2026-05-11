#include <iostream>
#include <vector>

void cpu_matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int N) {
    // Implementation for CPU matrix multiplication
    // Compute C = A * B on the CPU
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;

            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void print_matrix(const std::vector<float>& M , int N , const char* name) {
    std::cout << "Matrix " << name << ":\n";
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            std::cout << M[row * N + col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int N = 4;

    std::vector<float> A(N*N);
    std::vector<float> B(N*N);
    std::vector<float> C_cpu(N*N, 0.0f);

    // Initialize A and B with some values
    for (int i = 0; i < N*N; i++) {
        A[i] = static_cast<float>(i + 1); // A = [1, 2, ..., 16]
        B[i] = 1.0f; // B = [1, 1, ..., 1]
    }

    cpu_matmul(A, B, C_cpu, N);

    print_matrix(A, N, "A");
    print_matrix(B, N, "B");
    print_matrix(C_cpu, N, "C_cpu");

    return 0;
}