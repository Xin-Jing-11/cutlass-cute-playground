// Instantiates CUTLASS SGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A * B + beta * C, all matrices column-major.

#include "sgemm_naive.cuh"
#include "sgemm_smem.cuh"

#define INSTANTIATE_SGEMM_NAIVE(BLOCK)                                      \
  extern "C" void cutlass_sgemm_naive_##BLOCK(                              \
      int m, int n, int k,                                                 \
      float alpha,                                                         \
      const float* A, int ldA,                                             \
      const float* B, int ldB,                                             \
      float beta,                                                          \
      float* C, int ldC) {                                                 \
    sgemm_naive<BLOCK>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);     \
  }

INSTANTIATE_SGEMM_NAIVE(32)

#undef INSTANTIATE_SGEMM_NAIVE

#define INSTANTIATE_SGEMM_SMEM(BLOCK)                                       \
  extern "C" void cutlass_sgemm_smem_##BLOCK(                               \
      int m, int n, int k,                                                  \
      float alpha,                                                          \
      const float* A, int ldA,                                              \
      const float* B, int ldB,                                              \
      float beta,                                                           \
      float* C, int ldC) {                                                  \
    sgemm_smem<BLOCK>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);       \
  }

INSTANTIATE_SGEMM_SMEM(32)

#undef INSTANTIATE_SGEMM_SMEM
