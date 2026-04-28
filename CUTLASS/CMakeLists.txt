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
    # Take only the first GPU's compute capability (multi-GPU returns one line per device)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" GPU_CC_FIRST "${GPU_CC}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${GPU_CC_FIRST}")
    # For SM90 (Hopper), always use 90a to enable TMA + WGMMA
    if(CUDA_ARCH_LIST STREQUAL "90")
        set(CUDA_ARCH_LIST "90a")
    endif()
else()
    set(CUDA_ARCH_LIST "120")
endif()
message(STATUS "Detected GPU architecture: ${CUDA_ARCH_LIST}")

# Add sm_90a if we want WGMMA atoms to compile. Needed by hgemm_wgmma.
# Note: sm_90a-compiled code only *executes* on Hopper (H100/H200).
# On Blackwell consumer (SM120), the binary builds but the WGMMA kernel
# falls through to CUTE_INVALID_CONTROL_PATH at runtime.
if(NOT "90a" IN_LIST CUDA_ARCH_LIST)
    list(APPEND CUDA_ARCH_LIST "90a")
endif()

# Upgrade "120" -> "120a": hgemm_tma uses SM90 TMA intrinsics. The SM120
# (shared::cta) PTX variant is only emitted when CUTLASS_ARCH_MMA_SM120A_ENABLED
# is defined, which needs sm_120a (not plain sm_120). Without this, TMA hits
# CUTE_INVALID_CONTROL_PATH on SM120.
list(TRANSFORM CUDA_ARCH_LIST REPLACE "^120$" "120a")
list(REMOVE_DUPLICATES CUDA_ARCH_LIST)

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

# Shared library exposing C entry points for Python (ctypes)
add_library(cutlass_kernels SHARED ${KERNEL_SOURCES})
target_link_libraries(cutlass_kernels cublas cuda -static-libstdc++)
set_target_properties(cutlass_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
