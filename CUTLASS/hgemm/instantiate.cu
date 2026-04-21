// Instantiates CUTLASS HGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A^T * B + beta * C (TN layout), all matrices column-major.
// FP16 in/out, FP32 accumulator.

#include <cuda_fp16.h>
#include "hgemm_wmma.cuh"
#include "hgemm_multistage.cuh"
#include "hgemm_tma.cuh"

#define INSTANTIATE_HGEMM_WMMA(BM, BN, BK) \
  extern "C" void cutlass_hgemm_wmma_##BM##x##BN##x##BK( \
      int m, int n, int k,                               \
      float alpha,                                       \
      const half* A, int ldA,                            \
      const half* B, int ldB,                            \
      float beta,                                        \
      half* C, int ldC) {                                 \
    hgemm_wmma<BM, BN, BK>(                             \
        m, n, k, alpha,                                  \
        reinterpret_cast<const cute::half_t*>(A), ldA,   \
        reinterpret_cast<const cute::half_t*>(B), ldB,   \
        beta,                                            \
        reinterpret_cast<cute::half_t*>(C), ldC);        \
  }

INSTANTIATE_HGEMM_WMMA(64, 256, 32)

#undef INSTANTIATE_HGEMM_WMMA

#define INSTANTIATE_HGEMM_MULTISTAGE(BM, BN, BK, STAGES) \
  extern "C" void cutlass_hgemm_multistage_##BM##x##BN##x##BK##x##STAGES( \
      int m, int n, int k,                               \
      float alpha,                                       \
      const half* A, int ldA,                            \
      const half* B, int ldB,                            \
      float beta,                                        \
      half* C, int ldC) {                                 \
    hgemm_multistage<BM, BN, BK, STAGES>(               \
        m, n, k, alpha,                                  \
        reinterpret_cast<const cute::half_t*>(A), ldA,   \
        reinterpret_cast<const cute::half_t*>(B), ldB,   \
        beta,                                            \
        reinterpret_cast<cute::half_t*>(C), ldC);        \
  }

INSTANTIATE_HGEMM_MULTISTAGE(128, 128, 32, 3)

#undef INSTANTIATE_HGEMM_MULTISTAGE

#define INSTANTIATE_HGEMM_TMA(BM, BN, BK, STAGES) \
  extern "C" void cutlass_hgemm_tma_##BM##x##BN##x##BK##x##STAGES( \
      int m, int n, int k,                               \
      float alpha,                                       \
      const half* A, int ldA,                            \
      const half* B, int ldB,                            \
      float beta,                                        \
      half* C, int ldC) {                                 \
    hgemm_tma<BM, BN, BK, STAGES>(                      \
        m, n, k, alpha,                                  \
        reinterpret_cast<const cute::half_t*>(A), ldA,   \
        reinterpret_cast<const cute::half_t*>(B), ldB,   \
        beta,                                            \
        reinterpret_cast<cute::half_t*>(C), ldC);        \
  }

// autotune winner @ 4096 (warmup=20, iters=50): 100.6 TF/s, 94.3% cuBLAS
INSTANTIATE_HGEMM_TMA(128, 64, 64, 2)

#undef INSTANTIATE_HGEMM_TMA

