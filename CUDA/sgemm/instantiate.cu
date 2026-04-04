// Instantiates CUDA SGEMM kernels as extern "C" for Python ctypes.
// D = alpha * A * B + beta * C, all matrices row-major.

#include "sgemm_naive.cuh"

#define INSTANTIATE_SGEMM_NAIVE(BLOCK)                          \
  extern "C" void cuda_sgemm_naive_##BLOCK(                     \
      int M, int N, int K,                                     \
      float alpha,                                             \
      const float* A, const float* B,                          \
      float beta,                                              \
      const float* C, float* D) {                              \
    sgemm_naive<BLOCK>(M, N, K, alpha, A, B, beta, C, D);      \
  }

INSTANTIATE_SGEMM_NAIVE(32)

#undef INSTANTIATE_SGEMM_NAIVE
