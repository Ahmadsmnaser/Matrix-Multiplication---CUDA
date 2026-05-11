# cuBLAS Comparison

This document records the first benchmark comparison between the custom naive CUDA kernel, the tiled shared-memory CUDA kernel, and NVIDIA cuBLAS.

## Configuration

- CUDA device count: `1`
- Matrix sizes: `256 x 256`, `512 x 512`, `1024 x 1024`
- Tile and block size: `32 x 32`
- GPU timing iterations: `10`
- Verification: `PASSED` for all tested sizes and implementations

## Benchmark Table

| Implementation | N | CPU ms | CPU GFLOPS | GPU ms | GPU GFLOPS | Speedup | Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Naive | 256 | 104.198 | 0.322 | 0.500 | 67.120 | 208.431x | YES |
| Tiled | 256 | 100.181 | 0.335 | 0.232 | 144.927 | 432.699x | YES |
| cuBLAS | 256 | 100.719 | 0.333 | 0.061 | 551.650 | 1655.865x | YES |
| Naive | 512 | 1152.194 | 0.233 | 4.028 | 66.639 | 286.031x | YES |
| Tiled | 512 | 1361.877 | 0.197 | 1.811 | 148.263 | 752.196x | YES |
| cuBLAS | 512 | 1479.402 | 0.181 | 0.332 | 809.586 | 4461.794x | YES |
| Naive | 1024 | 11642.487 | 0.184 | 35.101 | 61.180 | 331.684x | YES |
| Tiled | 1024 | 10864.642 | 0.198 | 20.376 | 105.395 | 533.217x | YES |
| cuBLAS | 1024 | 11062.753 | 0.194 | 16.578 | 129.539 | 667.322x | YES |

## GPU-Only Comparison

| N | Naive GPU ms | Tiled GPU ms | cuBLAS GPU ms | Tiled vs Naive | cuBLAS vs Tiled | cuBLAS vs Naive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.500 | 0.232 | 0.061 | 2.155x | 3.803x | 8.197x |
| 512 | 4.028 | 1.811 | 0.332 | 2.224x | 5.455x | 12.133x |
| 1024 | 35.101 | 20.376 | 16.578 | 1.723x | 1.229x | 2.117x |

## Key Takeaways

- cuBLAS is the fastest implementation at every tested size.
- The custom tiled kernel gives a large improvement over the naive kernel, especially at `N=256` and `N=512`.
- At `N=1024`, cuBLAS is still faster than the tiled kernel, but the gap is much smaller than at the smaller sizes.
- The tiled kernel reaches `105.395 GFLOPS` at `N=1024`, while cuBLAS reaches `129.539 GFLOPS` for the same size.

## Notes

- CPU timings are recomputed for each implementation row, so small CPU timing differences are expected.
- cuBLAS uses a highly optimized production implementation, so it is the reference point for the custom CUDA kernels.
- These results complete the first project comparison milestone: CPU reference, naive CUDA, tiled CUDA, and cuBLAS.
