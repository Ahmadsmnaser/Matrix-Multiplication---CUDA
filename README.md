# Matrix Multiplication Project

This repository is organized for a CPU-first implementation that will later grow into a CUDA matrix multiplication project.

## Project Layout

- `include/` public headers
- `src/cpu/` CPU baseline implementation
- `src/cuda/` CUDA kernels and GPU-side code
- `src/common/` shared helpers and utilities
- `benchmarks/` timing and GFLOPS measurement programs
- `tests/` correctness tests
- `docs/` notes, profiling results, and project writeups

## Current Status

The project structure is in place, but no implementation has been added yet.

## Next Step

Start with the CPU baseline:

- matrix allocation
- matrix initialization
- CPU matrix multiplication
- correctness checks
- simple timing
