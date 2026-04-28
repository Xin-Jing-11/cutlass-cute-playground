// Instantiates CUDA HGEMM kernels as extern "C" for Python ctypes.
// C = alpha * A^T * B + beta * C  (TN layout), all matrices column-major.
// FP16 in/out, FP32 accumulator.

#include "hgemm_wgmma_tma.cuh"
#include "hgemm_warp_specialization.cuh"
#include "hgemm_persistent.cuh"
#include "hgemm_cluster.cuh"
#include "hgemm_epilogue.cuh"
#include "hgemm_optimized.cuh"

#define INSTANTIATE_HGEMM_WGMMA_TMA(BM, BN, BK, NWG)                              \
  extern "C" void cuda_hgemm_wgmma_tma_##BM##x##BN##x##BK##x##NWG(               \
      int M, int N, int K,                                                         \
      float alpha,                                                                 \
      const half* A, int ldA,                                                      \
      const half* B, int ldB,                                                      \
      float beta,                                                                  \
      half* C, int ldC) {                                                           \
    hgemm_wgmma_tma::hgemm_wgmma_tma<BM, BN, BK, NWG>(                            \
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);                             \
  }

#define INSTANTIATE_HGEMM_WARP_SPEC(BM, BN, BK, NCS, QS)                          \
  extern "C" void cuda_hgemm_warp_spec_##BM##x##BN##x##BK##x##NCS##x##QS(        \
      int M, int N, int K,                                                         \
      float alpha,                                                                 \
      const half* A, int ldA,                                                      \
      const half* B, int ldB,                                                      \
      float beta,                                                                  \
      half* C, int ldC) {                                                           \
    hgemm_warp_specialization::hgemm_warp_specialization<BM, BN, BK, NCS, QS>(              \
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);                             \
  }

#define INSTANTIATE_HGEMM_PERSISTENT(BM, BN, BK, NCS, QS)                           \
  extern "C" void cuda_hgemm_persistent_##BM##x##BN##x##BK##x##NCS##x##QS(        \
      int M, int N, int K,                                                         \
      float alpha,                                                                 \
      const half* A, int ldA,                                                      \
      const half* B, int ldB,                                                      \
      float beta,                                                                  \
      half* C, int ldC) {                                                           \
    hgemm_persistent::hgemm_persistent<BM, BN, BK, NCS, QS>(                      \
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);                             \
  }

// Autotune results at 4096³ on H100 NVL — fastest per family kept.

INSTANTIATE_HGEMM_WGMMA_TMA(128, 128, 64, 1)

// warp_spec: best = 128x256x64 NCS=2 QS=4 — 0.395ms 348 GF/s
INSTANTIATE_HGEMM_WARP_SPEC(128, 256, 64, 2, 4)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 128, 64, 1, 2)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 128, 64, 1, 3)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 128, 64, 1, 5)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 128, 64, 2, 3)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 256, 64, 2, 2)
// INSTANTIATE_HGEMM_WARP_SPEC(128, 256, 64, 2, 3)

// persistent: best = 128x256x64 NCS=2 QS=4 — 0.396ms 347 GF/s
INSTANTIATE_HGEMM_PERSISTENT(128, 256, 64, 2, 4)
// INSTANTIATE_HGEMM_PERSISTENT(128, 128, 64, 1, 3)
// INSTANTIATE_HGEMM_PERSISTENT(128, 128, 64, 1, 5)
// INSTANTIATE_HGEMM_PERSISTENT(128, 128, 64, 2, 3)
// INSTANTIATE_HGEMM_PERSISTENT(128, 256, 64, 2, 3)

#define INSTANTIATE_HGEMM_CLUSTER(BM, BN, BK, NCS, QS, CM, CN)                    \
  extern "C" void cuda_hgemm_cluster_##BM##x##BN##x##BK##x##NCS##x##QS##x##CM##x##CN( \
      int M, int N, int K,                                                         \
      float alpha,                                                                 \
      const half* A, int ldA,                                                      \
      const half* B, int ldB,                                                      \
      float beta,                                                                  \
      half* C, int ldC) {                                                           \
    hgemm_cluster::hgemm_cluster<BM, BN, BK, NCS, QS, CM, CN>(                    \
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);                             \
  }

// cluster: best = 128x256x64 NCS=2 QS=4 cluster(2,1) — 0.410ms 336 GF/s
INSTANTIATE_HGEMM_CLUSTER(128, 256, 64, 2, 4, 2, 1)
// INSTANTIATE_HGEMM_CLUSTER(128, 128, 64, 1, 3, 2, 1)
// INSTANTIATE_HGEMM_CLUSTER(128, 128, 64, 2, 3, 2, 1)
// INSTANTIATE_HGEMM_CLUSTER(128, 256, 64, 2, 3, 2, 1)

#define INSTANTIATE_HGEMM_EPILOGUE(BM, BN, BK, NCS, QS, CM, CN)                     \
  extern "C" void cuda_hgemm_epilogue_##BM##x##BN##x##BK##x##NCS##x##QS##x##CM##x##CN( \
      int M, int N, int K,                                                         \
      float alpha,                                                                 \
      const half* A, int ldA,                                                      \
      const half* B, int ldB,                                                      \
      float beta,                                                                  \
      half* C, int ldC) {                                                           \
    hgemm_epilogue::hgemm_epilogue<BM, BN, BK, NCS, QS, CM, CN>(                  \
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);                             \
  }

// epilogue: best = 128x256x64 NCS=2 QS=2 cluster(2,1) — 0.322ms 427 GF/s
INSTANTIATE_HGEMM_EPILOGUE(128, 256, 64, 2, 2, 2, 1)
// INSTANTIATE_HGEMM_EPILOGUE(128, 128, 64, 2, 3, 2, 1)
// INSTANTIATE_HGEMM_EPILOGUE(128, 256, 64, 2, 3, 2, 1)

#define INSTANTIATE_HGEMM_OPTIMIZED(BM, BN, BK, NCS, QS, CM, CN)                     \
  extern "C" void cuda_hgemm_optimized_##BM##x##BN##x##BK##x##NCS##x##QS##x##CM##x##CN( \
      int M, int N, int K,                                                         \
      float alpha,                                                                 \
      const half* A, int ldA,                                                      \
      const half* B, int ldB,                                                      \
      float beta,                                                                  \
      half* C, int ldC) {                                                           \
    hgemm_optimized::hgemm_optimized<BM, BN, BK, NCS, QS, CM, CN>(                \
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);                             \
  }

// optimized: PTX barriers + scaleD=0 + Hilbert curve + async TMA store
// Best configs from autotune sweep at 4096³ on H100 NVL (sorted by perf):
//   #1  128x256x64 NCS=2 QS=3 cluster(2,1)  — 0.302ms 455 GF/s
//   #2  128x256x64 NCS=2 QS=2 cluster(2,1)  — 0.308ms 446 GF/s
//   #3  128x128x64 NCS=2 QS=3 cluster(2,1)  — 0.355ms 387 GF/s
INSTANTIATE_HGEMM_OPTIMIZED(128, 256, 64, 2, 3, 2, 1)
// INSTANTIATE_HGEMM_OPTIMIZED(128, 128, 64, 2, 3, 2, 1)
// INSTANTIATE_HGEMM_OPTIMIZED(128, 256, 64, 2, 2, 2, 1)

#undef INSTANTIATE_HGEMM_WGMMA_TMA
#undef INSTANTIATE_HGEMM_WARP_SPEC
#undef INSTANTIATE_HGEMM_PERSISTENT
#undef INSTANTIATE_HGEMM_CLUSTER
#undef INSTANTIATE_HGEMM_EPILOGUE
#undef INSTANTIATE_HGEMM_OPTIMIZED


