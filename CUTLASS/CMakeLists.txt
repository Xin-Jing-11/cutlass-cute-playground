cmake_minimum_required(VERSION 3.18)
project(cutlass_gemm CUDA CXX)

# Auto-detect GPU architecture via nvidia-smi
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_CC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(GPU_CC)
    string(REPLACE "." "" CUDA_ARCH_LIST "${GPU_CC}")
else()
    set(CUDA_ARCH_LIST "120")
endif()
message(STATUS "Detected GPU architecture: ${CUDA_ARCH_LIST}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# CUTLASS header-only library
set(CUTLASS_HOME "$ENV{CUTLASS_HOME}")
if(NOT CUTLASS_HOME)
    set(CUTLASS_HOME "/usr/local/cutlass")
endif()
message(STATUS "Using CUTLASS from: ${CUTLASS_HOME}")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

include_directories(
    ${CUTLASS_HOME}/include
    ${CUTLASS_HOME}/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Collect instantiation files from kernel subdirectories
file(GLOB_RECURSE KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*/instantiate.cu)

# Shared library exposing C entry points for Python (ctypes)
add_library(cutlass_kernels SHARED ${KERNEL_SOURCES})
target_link_libraries(cutlass_kernels cublas -static-libstdc++)
set_target_properties(cutlass_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
