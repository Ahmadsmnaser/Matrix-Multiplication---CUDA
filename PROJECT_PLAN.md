# Matrix Multiplication — CUDA Project Plan

## Goal

Build a CUDA matrix multiplication project step by step, starting from a naive global-memory kernel and progressing toward a tiled shared-memory implementation, then compare the result against cuBLAS.

The main performance metric is GFLOPS:

```text
GFLOPS = (2 * N^3) / (time_seconds * 1e9)
```

For square `N x N` matrices, matrix multiplication performs approximately `2 * N^3` floating-point operations.

## Phase 1 — Naive Kernel

**Estimated time:** 2 days

Each CUDA thread computes one element of the output matrix:

```text
C[row][col] = dot(A[row], B[col])
```

### Tasks

- Implement a basic CUDA kernel where each thread computes one `C[row][col]`.
- Use global memory only.
- Add bounds checks for matrix sizes that are not exact multiples of the block size.
- Allocate and copy matrices between host and device.
- Measure kernel execution time.
- Compute GFLOPS as the baseline result.

### Expected result

This version should be correct but not highly optimized. It provides the baseline for all later comparisons.

## Phase 2 — Tiled Shared Memory

**Estimated time:** 3 days

Use shared memory to reduce repeated global memory reads.

### Idea

Partition the input matrices into tiles. Each thread block cooperatively loads one tile from `A` and one tile from `B` into shared memory, computes partial dot products, then moves to the next tile.

### Tasks

- Implement a tiled matrix multiplication kernel using shared memory.
- Synchronize threads after loading each tile.
- Experiment with tile sizes:
  - `16 x 16`
  - `32 x 32`
- Compare correctness against the naive kernel.
- Measure execution time and GFLOPS.

### Expected result

This is the core optimization phase. A tiled implementation should be significantly faster than the naive version, often around `5x` to `10x`, depending on matrix size, GPU, and implementation details.

## Phase 3 — cuBLAS Comparison

**Estimated time:** 1 day

Compare the custom kernels against NVIDIA's optimized cuBLAS implementation.

### Tasks

- Run `cublasSgemm` on the same matrix sizes and input data.
- Measure cuBLAS execution time.
- Compute cuBLAS GFLOPS.
- Compare:
  - Naive kernel
  - Tiled shared-memory kernel
  - cuBLAS

### Expected result

cuBLAS should be much faster than the custom kernels. The performance gap shows how close the tiled kernel is to a highly optimized production implementation.

## Phase 4 — Profiling and Polish

**Estimated time:** 2 days

Use profiling tools and document the final results.

### Tasks

- Profile kernels with Nsight Compute.
- Track:
  - Arithmetic intensity
  - Memory throughput
  - Occupancy
  - Shared memory usage
  - Global memory load efficiency
- Create a results table.
- Write a clear README.
- Push the project to GitHub.

### Results Table Template

| Implementation | Matrix Size | Time (ms) | GFLOPS | Speedup vs Naive |
| --- | ---: | ---: | ---: | ---: |
| Naive global memory | `N x N` | TBD | TBD | `1.00x` |
| Tiled shared memory | `N x N` | TBD | TBD | TBD |
| cuBLAS `cublasSgemm` | `N x N` | TBD | TBD | TBD |

## Final Deliverables

- Naive CUDA matrix multiplication kernel.
- Tiled shared-memory CUDA matrix multiplication kernel.
- cuBLAS benchmark.
- Correctness checks between implementations.
- Timing and GFLOPS measurements.
- Profiling notes from Nsight Compute.
- README with results and observations.

