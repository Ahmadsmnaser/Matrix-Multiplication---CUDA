# Naive CUDA Baseline

This document records the baseline benchmark results for the naive CUDA matrix multiplication implementation and its CPU reference comparison.

## Configuration

- Matrix sizes: `256 x 256`, `512 x 512`, `1024 x 1024`
- CUDA block size: `16 x 16`
- GPU timing iterations: `10`
- Verification: `PASSED` for all tested sizes

## Benchmark Table

| N | CPU ms | CPU GFLOPS | GPU ms | GPU GFLOPS | Speedup | Verified |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 256 | 106.172 | 0.316 | 0.486 | 69.058 | 218.512x | YES |
| 512 | 1004.581 | 0.267 | 4.407 | 60.913 | 227.957x | YES |
| 1024 | 13217.790 | 0.162 | 37.103 | 57.879 | 356.248x | YES |

## Summary

- The naive CUDA kernel is already dramatically faster than the CPU reference for all tested sizes.
- GPU performance stays in the `~58 to 69 GFLOPS` range across this sweep.
- CPU performance drops as matrix size grows, while GPU speedup increases from `218.512x` at `N=256` to `356.248x` at `N=1024`.

## Notes

- This result is the baseline for the naive CUDA kernel.
- The matrices are currently initialized with constant values, which is fine for a first run but should later be replaced with more varied data for stronger validation.
- These numbers should be used as the reference point for the future tiled shared-memory version and the cuBLAS comparison.
- Future documentation can extend this file or add new files for:
  - shared-memory tiled CUDA
  - cuBLAS comparison
  - profiling with Nsight Compute
