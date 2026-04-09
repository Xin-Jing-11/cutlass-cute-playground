// Instantiates CUTLASS SGEMM kernels as extern "C" for Python ctypes.
// TN: D = alpha * A^T * B + beta * C. A(K,M), B(K,N), C(M,N) all col-major.

#include "sgemm_naive.cuh"
#include "sgemm_smem.cuh"
#include "sgemm_tiling.cuh"
#include "sgemm_vectorize.cuh"

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

#if 1  // M-contiguous As layout variant
#define INSTANTIATE_SGEMM_SMEM_MC(BLOCK)                                       \
  extern "C" void cutlass_sgemm_smem_mc_##BLOCK(                               \
      int m, int n, int k,                                                      \
      float alpha,                                                              \
      const float* A, int ldA,                                                  \
      const float* B, int ldB,                                                  \
      float beta,                                                               \
      float* C, int ldC) {                                                      \
    sgemm_smem<BLOCK, true>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);    \
  }

INSTANTIATE_SGEMM_SMEM_MC(32)

#undef INSTANTIATE_SGEMM_SMEM_MC
#endif

#define INSTANTIATE_SGEMM_TILING(BM, BN, BK, TM, TN)                          \
  extern "C" void cutlass_sgemm_tiling_##BM##x##BN##x##BK##x##TM##x##TN(      \
      int m, int n, int k,                                                     \
      float alpha,                                                             \
      const float* A, int ldA,                                                 \
      const float* B, int ldB,                                                 \
      float beta,                                                              \
      float* C, int ldC) {                                                     \
    sgemm_tiling<BM, BN, BK, TM, TN>(m, n, k, alpha, A, ldA, B, ldB,         \
                                      beta, C, ldC);                           \
  }

INSTANTIATE_SGEMM_TILING(64, 64, 16, 8, 8)
INSTANTIATE_SGEMM_TILING(128, 128, 16, 8, 8)

#undef INSTANTIATE_SGEMM_TILING

#if 1  // M-contiguous As layout variant
#define INSTANTIATE_SGEMM_TILING_MC(BM, BN, BK, TM, TN)                          \
  extern "C" void cutlass_sgemm_tiling_mc_##BM##x##BN##x##BK##x##TM##x##TN(      \
      int m, int n, int k,                                                         \
      float alpha,                                                                 \
      const float* A, int ldA,                                                     \
      const float* B, int ldB,                                                     \
      float beta,                                                                  \
      float* C, int ldC) {                                                         \
    sgemm_tiling<BM, BN, BK, TM, TN, true>(m, n, k, alpha, A, ldA, B, ldB,      \
                                             beta, C, ldC);                        \
  }

INSTANTIATE_SGEMM_TILING_MC(64, 64, 16, 8, 8)
INSTANTIATE_SGEMM_TILING_MC(128, 128, 16, 8, 8)

#undef INSTANTIATE_SGEMM_TILING_MC
#endif

