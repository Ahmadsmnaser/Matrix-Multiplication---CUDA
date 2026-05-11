#pragma once

#include <vector>

void cpu_matmul(const std::vector<float> &A,
                const std::vector<float> &B,
                std::vector<float> &C,
                int N);

