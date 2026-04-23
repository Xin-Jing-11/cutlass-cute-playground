// Instantiates CUTLASS HGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A^T * B + beta * C (TN layout), all matrices column-major.
// FP16 in/out, FP32 accumulator.

#include <cuda_fp16.h>
#include "hgemm_wmma.cuh"
#include "hgemm_multistage.cuh"
#include "hgemm_tma.cuh"

#if defined(__CUDA_ARCH__) || defined(CUTLASS_ARCH_MMA_SM90A_ENABLED) || (__CUDACC_VER_MAJOR__ >= 12)
#include "hgemm_wgmma.cuh"
#endif

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

// Best config from autotuning at 4096³ on H100 NVL
INSTANTIATE_HGEMM_WMMA(128, 128, 16)

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

// Best config from autotuning at 4096³ on H100 NVL
INSTANTIATE_HGEMM_MULTISTAGE(128, 128, 32, 4)

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

// Best config from autotuning at 4096³ on H100 NVL
INSTANTIATE_HGEMM_TMA(128, 64, 64, 2)

#undef INSTANTIATE_HGEMM_TMA

#if defined(CUTLASS_ARCH_MMA_SM90A_ENABLED) || (__CUDACC_VER_MAJOR__ >= 12)

#define INSTANTIATE_HGEMM_WGMMA(BM, BN, BK, STAGES) \
  extern "C" void cutlass_hgemm_wgmma_##BM##x##BN##x##BK##x##STAGES( \
      int m, int n, int k,                               \
      float alpha,                                       \
      const half* A, int ldA,                            \
      const half* B, int ldB,                            \
      float beta,                                        \
      half* C, int ldC) {                                 \
    hgemm_wgmma<BM, BN, BK, STAGES>(                    \
        m, n, k, alpha,                                  \
        reinterpret_cast<const cute::half_t*>(A), ldA,   \
        reinterpret_cast<const cute::half_t*>(B), ldB,   \
        beta,                                            \
        reinterpret_cast<cute::half_t*>(C), ldC);        \
  }

// Best config from autotuning at 4096³ on H100 NVL
INSTANTIATE_HGEMM_WGMMA(128, 128, 64, 3)

#undef INSTANTIATE_HGEMM_WGMMA

#include "hgemm_wgmma_cluster.cuh"

#define INSTANTIATE_HGEMM_WGMMA_CLUSTER(BM, BN, BK, STAGES) \
  extern "C" void cutlass_hgemm_wgmma_cluster_##BM##x##BN##x##BK##x##STAGES( \
      int m, int n, int k,                               \
      float alpha,                                       \
      const half* A, int ldA,                            \
      const half* B, int ldB,                            \
      float beta,                                        \
      half* C, int ldC) {                                 \
    hgemm_wgmma_cluster<BM, BN, BK, STAGES>(            \
        m, n, k, alpha,                                  \
        reinterpret_cast<const cute::half_t*>(A), ldA,   \
        reinterpret_cast<const cute::half_t*>(B), ldB,   \
        beta,                                            \
        reinterpret_cast<cute::half_t*>(C), ldC);        \
  }

// 1x2 cluster along N, same tile as wgmma
// Best config from autotuning at 4096³ on H100 NVL: 128x128x64x5 (377k GF/s)
INSTANTIATE_HGEMM_WGMMA_CLUSTER(128, 128, 64, 3)
INSTANTIATE_HGEMM_WGMMA_CLUSTER(128, 128, 64, 4)
INSTANTIATE_HGEMM_WGMMA_CLUSTER(128, 128, 64, 5)

#undef INSTANTIATE_HGEMM_WGMMA_CLUSTER

#endif

