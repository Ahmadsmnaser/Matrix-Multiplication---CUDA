PROJECT := matrix-multiplication-cuda

.PHONY: help tree clean

help:
	@printf "Project: $(PROJECT)\n\n"
	@printf "Current status: structure only, no implementation yet.\n\n"
	@printf "Planned layout:\n"
	@printf "  include/        Public headers\n"
	@printf "  src/cpu/        CPU baseline implementation\n"
	@printf "  src/cuda/       CUDA kernels and wrappers\n"
	@printf "  src/common/     Shared utilities\n"
	@printf "  benchmarks/     Timing and GFLOPS benchmarks\n"
	@printf "  tests/          Correctness tests\n"
	@printf "  docs/           Notes, results, profiling\n\n"
	@printf "Next step: add the CPU baseline in src/cpu/ and a matching header in include/.\n"

tree:
	@printf ".\n"
	@printf "|-- Makefile\n"
	@printf "|-- PROJECT_PLAN.md\n"
	@printf "|-- docs/\n"
	@printf "|-- include/\n"
	@printf "|-- benchmarks/\n"
	@printf "|-- src/\n"
	@printf "|   |-- common/\n"
	@printf "|   |-- cpu/\n"
	@printf "|   `-- cuda/\n"
	@printf "`-- tests/\n"

clean:
	@printf "Nothing to clean yet.\n"
