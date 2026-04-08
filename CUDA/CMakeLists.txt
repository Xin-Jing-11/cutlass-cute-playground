cmake_minimum_required(VERSION 3.18)
project(cuda_kernels CUDA)

# Auto-detect GPU architecture via nvidia-smi
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_CC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(GPU_CC)
    # Handle multi-GPU: take only the first architecture
    string(REGEX MATCH "[0-9]+\\.[0-9]+" GPU_CC_FIRST "${GPU_CC}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${GPU_CC_FIRST}")
else()
    set(CUDA_ARCH_LIST "120")
endif()
message(STATUS "Detected GPU architecture: ${CUDA_ARCH_LIST}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Collect instantiation files from kernel subdirectories
file(GLOB_RECURSE KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*/instantiate.cu)

if(KERNEL_SOURCES)
    add_library(cuda_kernels SHARED ${KERNEL_SOURCES})
    target_link_libraries(cuda_kernels stdc++)
    set_target_properties(cuda_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
else()
    message(STATUS "No .cu sources found yet — skipping cuda_kernels target")
endif()
