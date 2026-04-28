// CUTLASS HGEMM kernel instantiations.
// C = alpha * A^T * B + beta * C  (TN layout), all matrices column-major.
// FP16 in/out, FP32 accumulator.
// Best configs from autotuning at 4096x4096x4096 on H100 NVL.

#include "hgemm_wgmma_tma.cuh"
#include "hgemm_warp_specialization.cuh"
#include "hgemm_persistent.cuh"
#include "hgemm_cluster.cuh"
#include "hgemm_epilogue.cuh"
#include "hgemm_optimized.cuh"

// ---------- wgmma_tma ----------
// Best: 128x128x64x1 (0.356 ms, 386K GF/s)
#define INSTANTIATE_HGEMM_WGMMA_TMA(BM, BN, BK, NWG)                                \
  extern "C" void cutlass_hgemm_wgmma_tma_##BM##x##BN##x##BK##x##NWG(               \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const half* A, int ldA,                                                        \
      const half* B, int ldB,                                                        \
      float beta,                                                                    \
      half* C, int ldC) {                                                             \
    hgemm_wgmma_tma<BM, BN, BK, NWG>(                                               \
        M, N, K, alpha,                                                              \
        (const cute::half_t*)A, ldA,                                                 \
        (const cute::half_t*)B, ldB,                                                 \
        beta, (cute::half_t*)C, ldC);                                                \
  }

INSTANTIATE_HGEMM_WGMMA_TMA(128, 128, 64, 1)
// INSTANTIATE_HGEMM_WGMMA_TMA(64, 256, 64, 1)   // 0.356 ms -- tied
// INSTANTIATE_HGEMM_WGMMA_TMA(128, 256, 64, 2)   // 0.567 ms

// ---------- warp_spec ----------
// Best: 128x256x64x2x3 (0.358 ms, 384K GF/s)
#define INSTANTIATE_HGEMM_WARP_SPEC(BM, BN, BK, NCS, QS)                            \
  extern "C" void cutlass_hgemm_warp_spec_##BM##x##BN##x##BK##x##NCS##x##QS(        \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const half* A, int ldA,                                                        \
      const half* B, int ldB,                                                        \
      float beta,                                                                    \
      half* C, int ldC) {                                                             \
    hgemm_warp_spec<BM, BN, BK, NCS, QS>(                                           \
        M, N, K, alpha,                                                              \
        (const cute::half_t*)A, ldA,                                                 \
        (const cute::half_t*)B, ldB,                                                 \
        beta, (cute::half_t*)C, ldC);                                                \
  }

INSTANTIATE_HGEMM_WARP_SPEC(128, 256, 64, 2, 3)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 256, 64, 2, 4)  // 0.361 ms
// INSTANTIATE_HGEMM_WARP_SPEC(128, 128, 64, 2, 4)  // 0.367 ms

// ---------- persistent ----------
// Best: 128x128x64x2x3 (0.316 ms, 434K GF/s)
#define INSTANTIATE_HGEMM_PERSISTENT(BM, BN, BK, NCS, QS)                           \
  extern "C" void cutlass_hgemm_persistent_##BM##x##BN##x##BK##x##NCS##x##QS(       \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const half* A, int ldA,                                                        \
      const half* B, int ldB,                                                        \
      float beta,                                                                    \
      half* C, int ldC) {                                                             \
    hgemm_persistent<BM, BN, BK, NCS, QS>(                                          \
        M, N, K, alpha,                                                              \
        (const cute::half_t*)A, ldA,                                                 \
        (const cute::half_t*)B, ldB,                                                 \
        beta, (cute::half_t*)C, ldC);                                                \
  }

INSTANTIATE_HGEMM_PERSISTENT(128, 128, 64, 2, 3)
// INSTANTIATE_HGEMM_PERSISTENT(128, 256, 64, 2, 3)  // 0.322 ms
// INSTANTIATE_HGEMM_PERSISTENT(128, 256, 64, 2, 4)  // 0.322 ms

// ---------- cluster ----------
// Best: 128x256x64x2x4x2x1 (0.300 ms, 457K GF/s)
#define INSTANTIATE_HGEMM_CLUSTER(BM, BN, BK, NCS, QS, CM, CN)                      \
  extern "C" void cutlass_hgemm_cluster_##BM##x##BN##x##BK##x##NCS##x##QS##x##CM##x##CN( \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const half* A, int ldA,                                                        \
      const half* B, int ldB,                                                        \
      float beta,                                                                    \
      half* C, int ldC) {                                                             \
    hgemm_cluster<BM, BN, BK, NCS, QS, CM, CN>(                                     \
        M, N, K, alpha,                                                              \
        (const cute::half_t*)A, ldA,                                                 \
        (const cute::half_t*)B, ldB,                                                 \
        beta, (cute::half_t*)C, ldC);                                                \
  }

INSTANTIATE_HGEMM_CLUSTER(128, 256, 64, 2, 4, 2, 1)
// INSTANTIATE_HGEMM_CLUSTER(128, 256, 64, 2, 3, 2, 1)  // 0.304 ms
// INSTANTIATE_HGEMM_CLUSTER(128, 256, 64, 2, 4, 1, 2)  // 0.309 ms

// ---------- epilogue ----------
// Best: 128x256x64x2x3x2x1 (0.300 ms, 458K GF/s)
#define INSTANTIATE_HGEMM_EPILOGUE(BM, BN, BK, NCS, QS, CM, CN)                     \
  extern "C" void cutlass_hgemm_epilogue_##BM##x##BN##x##BK##x##NCS##x##QS##x##CM##x##CN( \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const half* A, int ldA,                                                        \
      const half* B, int ldB,                                                        \
      float beta,                                                                    \
      half* C, int ldC) {                                                             \
    hgemm_epilogue<BM, BN, BK, NCS, QS, CM, CN>(                                    \
        M, N, K, alpha,                                                              \
        (const cute::half_t*)A, ldA,                                                 \
        (const cute::half_t*)B, ldB,                                                 \
        beta, (cute::half_t*)C, ldC);                                                \
  }

INSTANTIATE_HGEMM_EPILOGUE(128, 256, 64, 2, 3, 2, 1)
// INSTANTIATE_HGEMM_EPILOGUE(128, 256, 64, 2, 2, 2, 1)  // 0.329 ms
// INSTANTIATE_HGEMM_EPILOGUE(128, 128, 64, 2, 3, 2, 1)  // 0.344 ms

// ---------- optimized ----------
// Best: 128x256x64x2x3x2x1 (0.298 ms, 461K GF/s)
#define INSTANTIATE_HGEMM_OPTIMIZED(BM, BN, BK, NCS, QS, CM, CN)                    \
  extern "C" void cutlass_hgemm_optimized_##BM##x##BN##x##BK##x##NCS##x##QS##x##CM##x##CN( \
      int M, int N, int K,                                                           \
      float alpha,                                                                   \
      const half* A, int ldA,                                                        \
      const half* B, int ldB,                                                        \
      float beta,                                                                    \
      half* C, int ldC) {                                                             \
    hgemm_optimized<BM, BN, BK, NCS, QS, CM, CN>(                                   \
        M, N, K, alpha,                                                              \
        (const cute::half_t*)A, ldA,                                                 \
        (const cute::half_t*)B, ldB,                                                 \
        beta, (cute::half_t*)C, ldC);                                                \
  }

INSTANTIATE_HGEMM_OPTIMIZED(128, 256, 64, 2, 3, 2, 1)
// INSTANTIATE_HGEMM_OPTIMIZED(128, 256, 64, 2, 2, 2, 1)  // 0.306 ms
// INSTANTIATE_HGEMM_OPTIMIZED(128, 128, 64, 2, 3, 2, 1)  // 0.319 ms

#undef INSTANTIATE_HGEMM_WGMMA_TMA
#undef INSTANTIATE_HGEMM_WARP_SPEC
#undef INSTANTIATE_HGEMM_PERSISTENT
#undef INSTANTIATE_HGEMM_CLUSTER
#undef INSTANTIATE_HGEMM_EPILOGUE
#undef INSTANTIATE_HGEMM_OPTIMIZED
