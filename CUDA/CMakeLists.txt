cmake_minimum_required(VERSION 3.18)

# Auto-detect GPU architecture via nvidia-smi BEFORE project() so that
# CMAKE_CUDA_ARCHITECTURES is set before CMake's compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_CC
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GPU_CC)
        string(REGEX MATCH "[0-9]+\\.[0-9]+" GPU_CC_FIRST "${GPU_CC}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${GPU_CC_FIRST}")
    else()
        set(CUDA_ARCH_LIST "90")
    endif()
    # Upgrade to arch-accelerated variants so WGMMA / TMA PTX compiles.
    string(REGEX REPLACE "^90$" "90a" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    string(REGEX REPLACE "^120$" "120a" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Detected GPU architecture: ${CUDA_ARCH_LIST}")
endif()

project(cuda_kernels CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Collect instantiation files from kernel subdirectories
# Use -DKERNEL_TYPES="sgemm;hgemm" to build only specific types (default: all)
if(KERNEL_TYPES)
    set(KERNEL_SOURCES "")
    foreach(kt ${KERNEL_TYPES})
        file(GLOB _kt_src ${CMAKE_CURRENT_SOURCE_DIR}/${kt}/instantiate.cu)
        list(APPEND KERNEL_SOURCES ${_kt_src})
    endforeach()
else()
    file(GLOB_RECURSE KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*/instantiate.cu)
endif()

if(KERNEL_SOURCES)
    add_library(cuda_kernels SHARED ${KERNEL_SOURCES})
    # cuda (driver stub) is needed for cuTensorMapEncodeTiled used by hgemm_tma.
    target_link_libraries(cuda_kernels stdc++ cuda)
    set_target_properties(cuda_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
else()
    message(STATUS "No .cu sources found yet — skipping cuda_kernels target")
endif()
