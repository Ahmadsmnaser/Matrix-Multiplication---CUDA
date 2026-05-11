# Matrix Multiplication — CUDA Benchmark

A step-by-step CUDA matrix multiplication benchmark that compares four implementations:

- CPU reference implementation
- Naive CUDA kernel
- Tiled shared-memory CUDA kernel
- NVIDIA cuBLAS `cublasSgemm`

The goal of this project is to understand CUDA performance progression: start with a correct CPU baseline, implement a naive GPU kernel, optimize it using shared-memory tiling, and compare the result against NVIDIA's optimized cuBLAS library.

---

## Overview

Matrix multiplication is a classic GPU workload because it combines:

- large amounts of parallel work
- repeated memory access
- arithmetic-heavy computation
- opportunities for shared-memory optimization

For square matrices of size `N x N`, the operation is:

```text
C = A x B
```

Each output element is computed as:

```text
C[row][col] = dot(A[row], B[col])
```

The benchmark reports execution time and GFLOPS for each implementation.

---

## Implementations

| Implementation | Description |
|---|---|
| CPU | Single-threaded CPU reference used for correctness checking |
| Naive CUDA | Each CUDA thread computes one output element using global memory |
| Tiled CUDA | Uses shared memory tiles to reduce repeated global memory reads |
| cuBLAS | Uses NVIDIA's optimized `cublasSgemm` implementation |

---

## Current Status

Implemented:

- CPU matrix multiplication reference
- Naive CUDA matrix multiplication kernel
- Tiled CUDA shared-memory kernel
- cuBLAS `cublasSgemm` benchmark
- Correctness verification against CPU output
- CUDA event timing
- GFLOPS calculation
- Benchmark results for `N = 256`, `512`, and `1024`

---

## Build

Build from the repository root:

```bash
make
```

This produces:

```text
src/cuda/MM
```

The build uses `nvcc` and links against cuBLAS:

```bash
nvcc -O2 -std=c++14 -Iinclude \
    src/cuda/matmul.cu \
    src/cuda/matmul_kernels.cu \
    src/cpu/matmul.cpp \
    -lcublas \
    -o src/cuda/MM
```

---

## Run

Run from the repository root:

```bash
./src/cuda/MM
```

Example output:

```text
Implementation  N       CPU ms        CPU GFLOPS      GPU ms      GPU GFLOPS      Speedup     Verified
Naive           256     ...
Tiled           256     ...
cuBLAS          256     ...
```

---

## WSL CUDA Note

When running inside WSL, the CUDA driver library may need to be visible through `LD_LIBRARY_PATH`.

If the program reports:

```text
cudaGetDeviceCount failed: no CUDA-capable device is detected
```

but `nvidia-smi` works, run:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then run the benchmark again:

```bash
./src/cuda/MM
```

To make this permanent:

```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Clean

Remove the generated binary:

```bash
make clean
```

---

## Project Layout

```text
.
├── Makefile
├── PROJECT_PLAN.md
├── README.md
├── docs/
│   ├── naive_cuda_baseline.md
│   ├── tiled_cuda_comparison.md
│   └── cublas_comparison.md
├── include/
│   ├── matmul_cpu.hpp
│   └── matmul_kernels.cuh
└── src/
    ├── cpu/
    │   └── matmul.cpp
    └── cuda/
        ├── matmul.cu
        └── matmul_kernels.cu
```

---

## Source Files

| File | Purpose |
|---|---|
| `include/matmul_cpu.hpp` | Declares the CPU reference function |
| `src/cpu/matmul.cpp` | Implements CPU matrix multiplication |
| `include/matmul_kernels.cuh` | Declares CUDA kernel launchers and tile configuration |
| `src/cuda/matmul_kernels.cu` | Implements naive and tiled CUDA kernels |
| `src/cuda/matmul.cu` | Benchmark driver for CPU, CUDA kernels, and cuBLAS |

---

## GFLOPS Formula

For square matrix multiplication:

```text
C = A x B
```

Each output element performs approximately:

```text
N multiplications + N additions
```

So the total floating-point operation count is approximately:

```text
2 * N^3
```

GFLOPS is computed as:

```text
GFLOPS = (2 * N^3) / (time_seconds * 1e9)
```

---

## Kernel Details

### Naive CUDA Kernel

Each CUDA thread computes one element of `C`:

```cpp
C[row * N + col] = sum(A[row * N + k] * B[k * N + col]);
```

This version uses global memory only.

It is simple and correct, but not highly optimized because many threads repeatedly load the same data from global memory.

---

### Tiled Shared-Memory Kernel

The tiled kernel divides the input matrices into `TILE_SIZE x TILE_SIZE` blocks.

Each thread block cooperatively loads:

- one tile from `A`
- one tile from `B`

into shared memory.

Threads then reuse these shared-memory tiles to compute partial dot products.

This reduces repeated global-memory reads and improves performance.

Current tile size:

```text
32 x 32
```

---

### cuBLAS

cuBLAS is NVIDIA's highly optimized BLAS library.

This project uses:

```cpp
cublasSgemm
```

Because cuBLAS assumes column-major matrix layout, while this project stores matrices in row-major order, the benchmark swaps `A` and `B` in the `cublasSgemm` call to produce the correct row-major result.

---

## Latest Results

Hardware:

```text
GPU: NVIDIA GeForce MX250
CUDA device count: 1
Tile/block size: 32 x 32
Iterations: 10
Matrix sizes: 256, 512, 1024
```

Latest documented run:

| Implementation | N | GPU Time (ms) | GPU GFLOPS | Verified |
|---|---:|---:|---:|:---:|
| Naive | 256 | 0.500 | 67.120 | YES |
| Tiled | 256 | 0.232 | 144.927 | YES |
| cuBLAS | 256 | 0.061 | 551.650 | YES |
| Naive | 512 | 4.028 | 66.639 | YES |
| Tiled | 512 | 1.811 | 148.263 | YES |
| cuBLAS | 512 | 0.332 | 809.586 | YES |
| Naive | 1024 | 35.101 | 61.180 | YES |
| Tiled | 1024 | 20.376 | 105.395 | YES |
| cuBLAS | 1024 | 16.578 | 129.539 | YES |

---

## Performance Summary

For `N = 512`:

| Implementation | GFLOPS | Relative to Naive |
|---|---:|---:|
| Naive CUDA | 66.639 | 1.00x |
| Tiled CUDA | 148.263 | 2.23x |
| cuBLAS | 809.586 | 12.15x |

The tiled shared-memory kernel is significantly faster than the naive global-memory kernel.

cuBLAS is much faster than both custom kernels, especially for `N = 256` and `N = 512`.

For `N = 1024`, cuBLAS is still faster than the tiled kernel, but the measured result is lower than expected compared to the `N = 512` result. This may be affected by laptop power limits, thermal throttling, or repeated CPU reference runs during benchmarking.

---

## Correctness

Matrices are currently initialized with:

```cpp
A[i] = 1.0f;
B[i] = 1.0f;
```

For this input, every output element should equal:

```text
N
```

Each GPU implementation is verified against the CPU reference result.

Current status:

```text
Naive CUDA: PASSED
Tiled CUDA: PASSED
cuBLAS: PASSED
```

---

## Notes

- Timing uses CUDA events.
- `cudaMalloc`, host-device copies, setup, and cleanup are not included in GPU timing.
- The CPU implementation is single-threaded and used mainly for correctness checking.
- CPU timing can vary because the CPU reference is currently recomputed for each benchmark row.
- The main performance comparison is between Naive CUDA, Tiled CUDA, and cuBLAS.
- Results were collected on a laptop GPU, so power and thermal limits can affect performance.

---

## Future Work

Planned improvements:

- Compute CPU reference only once per matrix size.
- Add speedup vs Naive CUDA directly in the output table.
- Add command-line arguments:
  - `--size`
  - `--iterations`
  - `--tile-size`
  - `--implementation`
- Add CSV export.
- Add Nsight Compute profiling notes.
- Compare tile sizes `16 x 16` and `32 x 32` more formally.
- Add support for random input matrices.
- Add cuBLAS-only benchmark mode for large matrix sizes.
- Add plots for GFLOPS vs matrix size.

---

## Lessons Learned

This project demonstrates several important CUDA performance ideas:

- Moving computation to the GPU can provide a large speedup over a simple CPU baseline.
- A naive CUDA kernel is correct but leaves performance on the table.
- Shared-memory tiling improves performance by reusing data within a thread block.
- cuBLAS is much faster than simple custom kernels because it uses highly optimized production-level implementations.
- Memory layout matters when calling libraries such as cuBLAS.
- Benchmark methodology matters: setup time, CPU reference time, and thermal behavior can all affect results.
