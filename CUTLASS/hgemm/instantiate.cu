// Instantiates CUTLASS HGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A^T * B + beta * C (TN layout), all matrices column-major.
// FP16 in/out, FP32 accumulator.

#include <cuda_fp16.h>
#include "hgemm_mma.cuh"

#define INSTANTIATE_HGEMM_MMA(BM, BN, BK) \
  extern "C" void cutlass_hgemm_mma_##BM##x##BN##x##BK( \
      int m, int n, int k,                               \
      float alpha,                                       \
      const half* A, int ldA,                            \
      const half* B, int ldB,                            \
      float beta,                                        \
      half* C, int ldC) {                                 \
    hgemm_mma<BM, BN, BK>(                              \
        m, n, k, alpha,                                  \
        reinterpret_cast<const cute::half_t*>(A), ldA,   \
        reinterpret_cast<const cute::half_t*>(B), ldB,   \
        beta,                                            \
        reinterpret_cast<cute::half_t*>(C), ldC);        \
  }

INSTANTIATE_HGEMM_MMA(128, 128, 32)

#undef INSTANTIATE_HGEMM_MMA
