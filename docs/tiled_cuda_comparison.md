# Tiled CUDA Comparison

This document records the first benchmark comparison between the naive CUDA matrix multiplication kernel and the tiled shared-memory CUDA kernel.

## Configuration

- Matrix sizes: `256 x 256`, `512 x 512`, `1024 x 1024`
- Tile and block size: `16 x 16`
- GPU timing iterations: `10`
- Verification: `PASSED` for all tested sizes and both implementations

## Benchmark Table

| Implementation | N | CPU ms | CPU GFLOPS | GPU ms | GPU GFLOPS | Speedup | Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Naive | 256 | 136.575 | 0.246 | 0.484 | 69.336 | 282.214x | YES |
| Tiled | 256 | 150.498 | 0.223 | 0.240 | 139.676 | 626.472x | YES |
| Naive | 512 | 1349.673 | 0.199 | 4.391 | 61.136 | 307.386x | YES |
| Tiled | 512 | 1306.796 | 0.205 | 1.891 | 141.945 | 691.016x | YES |
| Naive | 1024 | 12532.186 | 0.171 | 37.170 | 57.775 | 337.162x | YES |
| Tiled | 1024 | 11826.326 | 0.182 | 15.699 | 136.792 | 753.325x | YES |

## Key Takeaways

- The tiled shared-memory kernel is substantially faster than the naive kernel at every tested size.
- GPU performance improves from about `58 to 69 GFLOPS` in the naive version to about `137 to 142 GFLOPS` in the tiled version.
- The tiled kernel is roughly `2.0x` to `2.3x` faster than the naive GPU kernel in this benchmark sweep.
- End-to-end GPU speedup over the CPU reference grows to more than `750x` at `N=1024`.

## Per-Size GPU Comparison

| N | Naive GPU ms | Tiled GPU ms | Tiled vs Naive Speedup | Naive GPU GFLOPS | Tiled GPU GFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.484 | 0.240 | 2.017x | 69.336 | 139.676 |
| 512 | 4.391 | 1.891 | 2.322x | 61.136 | 141.945 |
| 1024 | 37.170 | 15.699 | 2.368x | 57.775 | 136.792 |

## Notes

- The CPU reference is still being recomputed for each benchmark row, so small CPU timing differences between naive and tiled rows are expected.
- The matrices are still initialized with constant values. That is acceptable for a first performance comparison, but more varied input data will make future correctness checks stronger.
- These results establish the first optimization milestone for the project and should be used as the baseline before comparing against cuBLAS.
