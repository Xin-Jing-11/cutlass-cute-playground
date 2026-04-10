// Instantiates CUDA SGEMM kernels as extern "C" for Python ctypes.
// TN: D = alpha * A^T * B + beta * C. A(K,M), B(K,N), C(M,N) all col-major.

#include "sgemm_naive.cuh"
#include "sgemm_smem.cuh"
#include "sgemm_tiling.cuh"
#include "sgemm_tiling_vectorize.cuh"
#include "sgemm_warptiling.cuh"
#include "sgemm_warptiling_vectorize.cuh"

#define INSTANTIATE_SGEMM_NAIVE(BLOCK)                          \
  extern "C" void cuda_sgemm_naive_##BLOCK(                     \
      int M, int N, int K,                                     \
      float alpha,                                             \
      const float* A, const float* B,                          \
      float beta,                                              \
      float* C) {                                               \
    sgemm_naive<BLOCK>(M, N, K, alpha, A, B, beta, C);         \
  }

// INSTANTIATE_SGEMM_NAIVE(32)

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

#define INSTANTIATE_SGEMM_SMEM_MC(BLOCK)                           \
  extern "C" void cuda_sgemm_smem_mc_##BLOCK(                      \
      int M, int N, int K,                                         \
      float alpha,                                                 \
      const float* A, const float* B,                              \
      float beta,                                                  \
      float* C) {                                                   \
    sgemm_smem<BLOCK, true>(M, N, K, alpha, A, B, beta, C);       \
  }

INSTANTIATE_SGEMM_SMEM_MC(32)

#undef INSTANTIATE_SGEMM_SMEM_MC

#define INSTANTIATE_SGEMM_TILING(BM, BN, BK, TM, TN)                        \
  extern "C" void cuda_sgemm_tiling_##BM##x##BN##x##BK##x##TM##x##TN(     \
      int M, int N, int K,                                                 \
      float alpha,                                                         \
      const float* A, const float* B,                                      \
      float beta,                                                          \
      float* C) {                                                           \
    sgemm_tiling<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta, C);     \
  }

INSTANTIATE_SGEMM_TILING(64, 64, 16, 8, 8)
INSTANTIATE_SGEMM_TILING(128, 128, 16, 8, 8)

#undef INSTANTIATE_SGEMM_TILING

#define INSTANTIATE_SGEMM_TILING_MC(BM, BN, BK, TM, TN)                          \
  extern "C" void cuda_sgemm_tiling_mc_##BM##x##BN##x##BK##x##TM##x##TN(       \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_tiling<BM, BN, BK, TM, TN, true>(M, N, K, alpha, A, B, beta, C);   \
  }

INSTANTIATE_SGEMM_TILING_MC(64, 64, 16, 8, 8)
INSTANTIATE_SGEMM_TILING_MC(128, 128, 16, 8, 8)

#undef INSTANTIATE_SGEMM_TILING_MC

#define INSTANTIATE_SGEMM_TILING_VECTORIZE(BM, BN, BK, TM, TN)                          \
  extern "C" void cuda_sgemm_tiling_vectorize_##BM##x##BN##x##BK##x##TM##x##TN(       \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_tiling_vectorize<BM, BN, BK, TM, TN>(M, N, K, alpha, A, B, beta, C);       \
  }

INSTANTIATE_SGEMM_TILING_VECTORIZE(64, 64, 16, 8, 8)
INSTANTIATE_SGEMM_TILING_VECTORIZE(128, 128, 16, 8, 8)

#undef INSTANTIATE_SGEMM_TILING_VECTORIZE

#define INSTANTIATE_SGEMM_TILING_VECTORIZE_MC(BM, BN, BK, TM, TN)                          \
  extern "C" void cuda_sgemm_tiling_vectorize_mc_##BM##x##BN##x##BK##x##TM##x##TN(       \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const float* A, const float* B,                                                \
      float beta,                                                                    \
      float* C) {                                                                     \
    sgemm_tiling_vectorize<BM, BN, BK, TM, TN, true>(M, N, K, alpha, A, B, beta, C);     \
  }

INSTANTIATE_SGEMM_TILING_VECTORIZE_MC(64, 64, 16, 8, 8)
INSTANTIATE_SGEMM_TILING_VECTORIZE_MC(128, 128, 16, 8, 8)

#undef INSTANTIATE_SGEMM_TILING_VECTORIZE_MC

#define INSTANTIATE_SGEMM_WARPTILING_VECTORIZE(BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN) \
  extern "C" void cuda_sgemm_warptiling_vectorize_##BM##x##BN##x##BK##x##WM##x##WN##x##WMITER##x##WNITER##x##TM##x##TN( \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_warptiling_vectorize<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>(    \
        M, N, K, alpha, A, B, beta, C);                                          \
  }

INSTANTIATE_SGEMM_WARPTILING_VECTORIZE(128, 128, 16, 64, 64, 1, 4, 8, 4)

#undef INSTANTIATE_SGEMM_WARPTILING_VECTORIZE

#if 1  // M-contiguous As layout variant
#define INSTANTIATE_SGEMM_WARPTILING_VECTORIZE_MC(BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN) \
  extern "C" void cuda_sgemm_warptiling_vectorize_mc_##BM##x##BN##x##BK##x##WM##x##WN##x##WMITER##x##WNITER##x##TM##x##TN( \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_warptiling_vectorize<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, true>( \
        M, N, K, alpha, A, B, beta, C);                                          \
  }

INSTANTIATE_SGEMM_WARPTILING_VECTORIZE_MC(128, 128, 16, 64, 64, 1, 4, 8, 4)

#undef INSTANTIATE_SGEMM_WARPTILING_VECTORIZE_MC
#endif

#define INSTANTIATE_SGEMM_WARPTILING(BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN) \
  extern "C" void cuda_sgemm_warptiling_##BM##x##BN##x##BK##x##WM##x##WN##x##WMITER##x##WNITER##x##TM##x##TN( \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_warptiling<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>(              \
        M, N, K, alpha, A, B, beta, C);                                          \
  }

INSTANTIATE_SGEMM_WARPTILING(128, 128, 16, 64, 64, 1, 4, 8, 4)

#undef INSTANTIATE_SGEMM_WARPTILING

#define INSTANTIATE_SGEMM_WARPTILING_MC(BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN) \
  extern "C" void cuda_sgemm_warptiling_mc_##BM##x##BN##x##BK##x##WM##x##WN##x##WMITER##x##WNITER##x##TM##x##TN( \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_warptiling<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, true>(        \
        M, N, K, alpha, A, B, beta, C);                                          \
  }

INSTANTIATE_SGEMM_WARPTILING_MC(128, 128, 16, 64, 64, 1, 4, 8, 4)

#undef INSTANTIATE_SGEMM_WARPTILING_MC