// Instantiates CUDA SGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A * B + beta * C, all matrices column-major.

#include "sgemm_naive.cuh"
#include "sgemm_smem.cuh"

#define INSTANTIATE_SGEMM_NAIVE(BLOCK)                          \
  extern "C" void cuda_sgemm_naive_##BLOCK(                     \
      int M, int N, int K,                                     \
      float alpha,                                             \
      const float* A, const float* B,                          \
      float beta,                                              \
      float* C) {                                               \
    sgemm_naive<BLOCK>(M, N, K, alpha, A, B, beta, C);         \
  }

INSTANTIATE_SGEMM_NAIVE(32)

#undef INSTANTIATE_SGEMM_NAIVE

#define INSTANTIATE_SGEMM_SMEM(BLOCK)                           \
  extern "C" void cuda_sgemm_smem_##BLOCK(                      \
      int M, int N, int K,                                     \
      float alpha,                                             \
      const float* A, const float* B,                          \
      float beta,                                              \
      float* C) {                                               \
    sgemm_smem<BLOCK>(M, N, K, alpha, A, B, beta, C);          \
  }

INSTANTIATE_SGEMM_SMEM(32)

#undef INSTANTIATE_SGEMM_SMEM
