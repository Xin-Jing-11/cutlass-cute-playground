.PHONY: build build-cutlass build-cuda clean-build clean-dsl-cache clean-pycache clean-all

build: build-cutlass build-cuda

build-cutlass:
	cd CUTLASS && mkdir -p build && cd build && cmake .. && make -j$$(nproc)

build-cuda:
	cd CUDA && mkdir -p build && cd build && cmake .. && make -j$$(nproc)

clean-cutlass:
	rm -rf CUTLASS/build

clean-cuda:
	rm -rf CUDA/build

clean-dsl-cache:
	rm -rf ~/.cutlass_cache

clean-pycache:
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-all: clean-cutlass clean-cuda clean-dsl-cache clean-pycache
