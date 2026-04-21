// Instantiates CUDA HGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A^T * B + beta * C  (TN layout), all matrices column-major.
// FP16 in/out, FP32 accumulator.

#include <cuda_fp16.h>
#include "hgemm_wmma.cuh"
#include "hgemm_multistage.cuh"
#include "hgemm_tma.cuh"

#define INSTANTIATE_HGEMM_WMMA(BM, BN, BK, WM, WN) \
  extern "C" void cuda_hgemm_wmma_##BM##x##BN##x##BK##x##WM##x##WN( \
      int m, int n, int k,                                           \
      float alpha,                                                   \
      const half* A, int ldA,                                        \
      const half* B, int ldB,                                        \
      float beta,                                                    \
      half* C, int ldC) {                                             \
    cuda_hgemm_wmma::hgemm_wmma<BM, BN, BK, WM, WN>(                  \
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);               \
  }

// autotune winner @ 4096: 88.8 TF/s (82.5% cuBLAS)
INSTANTIATE_HGEMM_WMMA(64, 64, 32, 2, 4)

#undef INSTANTIATE_HGEMM_WMMA


#define INSTANTIATE_HGEMM_MULTISTAGE(BM, BN, BK, WM, WN, STAGES) \
  extern "C" void cuda_hgemm_multistage_##BM##x##BN##x##BK##x##WM##x##WN##x##STAGES( \
      int m, int n, int k,                                                            \
      float alpha,                                                                    \
      const half* A, int ldA,                                                         \
      const half* B, int ldB,                                                         \
      float beta,                                                                     \
      half* C, int ldC) {                                                              \
    cuda_hgemm_multistage::hgemm_multistage<BM, BN, BK, WM, WN, STAGES>(               \
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);                                \
  }

// autotune winner @ 4096: 92.5 TF/s (88.6% cuBLAS)
INSTANTIATE_HGEMM_MULTISTAGE(128, 128, 32, 4, 2, 2)

#undef INSTANTIATE_HGEMM_MULTISTAGE


#define INSTANTIATE_HGEMM_TMA(BM, BN, BK, WM, WN, STAGES) \
  extern "C" void cuda_hgemm_tma_##BM##x##BN##x##BK##x##WM##x##WN##x##STAGES( \
      int m, int n, int k,                                                     \
      float alpha,                                                             \
      const half* A, int ldA,                                                  \
      const half* B, int ldB,                                                  \
      float beta,                                                              \
      half* C, int ldC) {                                                       \
    cuda_hgemm_tma::hgemm_tma<BM, BN, BK, WM, WN, STAGES>(                      \
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);                         \
  }

// autotune winner @ 4096 (STAGES in {2,4}): 96.7 TF/s (88.4% cuBLAS)
INSTANTIATE_HGEMM_TMA(64, 128, 32, 2, 4, 2)

#undef INSTANTIATE_HGEMM_TMA
