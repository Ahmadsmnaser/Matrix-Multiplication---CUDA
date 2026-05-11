NVCC := nvcc
CXXFLAGS := -O2 -std=c++14
INCLUDES := -Iinclude
LDFLAGS := -lcublas

TARGET := src/cuda/MM
SOURCES := src/cuda/matmul.cu src/cuda/matmul_kernels.cu src/cpu/matmul.cpp

.PHONY: all help tree clean

all: $(TARGET)

$(TARGET): $(SOURCES) include/matmul_cpu.hpp include/matmul_kernels.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LDFLAGS) -o $(TARGET)

help:
	@printf "Targets:\n"
	@printf "  make        Build the CUDA benchmark\n"
	@printf "  make clean  Remove generated binaries\n"
	@printf "  make tree   Print the project layout\n"

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
	rm -f $(TARGET)
