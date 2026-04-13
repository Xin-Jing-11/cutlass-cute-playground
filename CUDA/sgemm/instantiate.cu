// Instantiates CUDA SGEMM kernels as extern "C" for Python ctypes.
// TN: D = alpha * A^T * B + beta * C. A(K,M), B(K,N), C(M,N) all col-major.
// All kernels use M-contiguous As layout.

#include "sgemm_naive.cuh"
#include "sgemm_smem.cuh"
#include "sgemm_tiling.cuh"
#include "sgemm_tiling_vectorize.cuh"
#include "sgemm_warptiling.cuh"
#include "sgemm_warptiling_vectorize.cuh"
#include "sgemm_double_buffering.cuh"

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

// Default double_buffering: MC layout (KC=false), zero register spills.
// KC=true has precomputed smem bases (eliminates LOP3) but causes register
// spills on this register-saturated kernel, making it slower for 1x4.
// KC=true is beneficial for 2x2 (see _mc variants below for comparison).
#define INSTANTIATE_SGEMM_DOUBLE_BUFFERING(BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN) \
  extern "C" void cuda_sgemm_double_buffering_##BM##x##BN##x##BK##x##WM##x##WN##x##WMITER##x##WNITER##x##TM##x##TN( \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_double_buffering<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, false>( \
        M, N, K, alpha, A, B, beta, C);                                          \
  }

INSTANTIATE_SGEMM_DOUBLE_BUFFERING(128, 128, 16, 64, 64, 1, 4, 8, 4)
INSTANTIATE_SGEMM_DOUBLE_BUFFERING(128, 128, 16, 64, 64, 2, 2, 8, 4)

#undef INSTANTIATE_SGEMM_DOUBLE_BUFFERING

// KC variant: K-contiguous As, Swizzle<B,2,S>, 128-bit g2s, precomputed smem bases.
// Eliminates per-load LOP3 from s2r but incurs register spills (kernel is at the
// 255-register limit). Net result: faster for 2x2 (LOP3 savings win), slightly
// slower for 1x4 (spill cost wins).
#define INSTANTIATE_SGEMM_DOUBLE_BUFFERING_MC(BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN) \
  extern "C" void cuda_sgemm_double_buffering_mc_##BM##x##BN##x##BK##x##WM##x##WN##x##WMITER##x##WNITER##x##TM##x##TN( \
      int M, int N, int K,                                                       \
      float alpha,                                                               \
      const float* A, const float* B,                                            \
      float beta,                                                                \
      float* C) {                                                                 \
    sgemm_double_buffering<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, true>(  \
        M, N, K, alpha, A, B, beta, C);                                          \
  }

INSTANTIATE_SGEMM_DOUBLE_BUFFERING_MC(128, 128, 16, 64, 64, 1, 4, 8, 4)
INSTANTIATE_SGEMM_DOUBLE_BUFFERING_MC(128, 128, 16, 64, 64, 2, 2, 8, 4)

#undef INSTANTIATE_SGEMM_DOUBLE_BUFFERING_MC
