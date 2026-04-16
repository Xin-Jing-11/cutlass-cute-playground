.PHONY: build build-cutlass build-cuda clean-build clean-dsl-cache clean-pycache clean-all \
       build-cutlass-sgemm build-cutlass-hgemm build-cuda-sgemm build-cuda-hgemm \
       build-sgemm build-hgemm clean-cutlass clean-cuda \
       clean-cutlass-sgemm clean-cutlass-hgemm clean-cuda-sgemm clean-cuda-hgemm \
       clean-sgemm clean-hgemm

build: build-cutlass build-cuda

# --- Build all kernels ---
build-cutlass:
	cd CUTLASS && mkdir -p build && cd build && cmake .. && make -j$$(nproc)

build-cuda:
	cd CUDA && mkdir -p build && cd build && cmake .. && make -j$$(nproc)

# --- Build by kernel type (both CUTLASS + CUDA) ---
build-sgemm: build-cutlass-sgemm build-cuda-sgemm

build-hgemm: build-cutlass-hgemm build-cuda-hgemm

# --- Build by kernel type + framework ---
build-cutlass-sgemm:
	cd CUTLASS && mkdir -p build && cd build && cmake .. -DKERNEL_TYPES=sgemm && make -j$$(nproc)

build-cutlass-hgemm:
	cd CUTLASS && mkdir -p build && cd build && cmake .. -DKERNEL_TYPES=hgemm && make -j$$(nproc)

build-cuda-sgemm:
	cd CUDA && mkdir -p build && cd build && cmake .. -DKERNEL_TYPES=sgemm && make -j$$(nproc)

build-cuda-hgemm:
	cd CUDA && mkdir -p build && cd build && cmake .. -DKERNEL_TYPES=hgemm && make -j$$(nproc)

# --- Clean ---
clean-cutlass:
	rm -rf CUTLASS/build

clean-cuda:
	rm -rf CUDA/build

clean-build: clean-cutlass clean-cuda

# --- Clean by kernel type ---
clean-cutlass-sgemm:
	rm -rf CUTLASS/build/CMakeFiles/cutlass_kernels.dir/sgemm

clean-cutlass-hgemm:
	rm -rf CUTLASS/build/CMakeFiles/cutlass_kernels.dir/hgemm

clean-cuda-sgemm:
	rm -rf CUDA/build/CMakeFiles/cuda_kernels.dir/sgemm

clean-cuda-hgemm:
	rm -rf CUDA/build/CMakeFiles/cuda_kernels.dir/hgemm

clean-sgemm: clean-cutlass-sgemm clean-cuda-sgemm

clean-hgemm: clean-cutlass-hgemm clean-cuda-hgemm

clean-dsl-cache:
	rm -rf ~/.cutlass_cache

clean-pycache:
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-all: clean-cutlass clean-cuda clean-dsl-cache clean-pycache
