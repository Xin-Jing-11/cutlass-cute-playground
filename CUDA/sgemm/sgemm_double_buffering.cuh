#pragma once
#include <cassert>
#include <utility>
#include <cuda/pipeline>
#include "../share.cuh"

/*
 * DOUBLE_BUFFERING SGEMM (TN): C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * Uses cuda::memcpy_async + cuda::pipeline API for gmem->smem async copies.
 *
 * BM, BN, BK: tile size for M, N, K dimension
 * WM, WN: warp tile size for M, N, dimension
 * WMITER, WNITER: number of subtiles for each warp on M, N dimension
 * TM, TN: tile size for thread local register
 *
 * M-contiguous As layout: As[k*BM + m].
 */

namespace db {

// -----------------------------------------------------------------------------
// gmem → smem (async): scalar per-float cp.async for A so we can keep
// per-element swizzle that eliminates smem bank conflicts on smem→reg.
// -----------------------------------------------------------------------------
template <int BM, int BK, int NUM_THREADS, int SWZ_B, int SWZ_S>
__device__ __forceinline__ void load_A_gmem_to_smem(
    const float* __restrict__ A, int K, float* As, int tid,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    constexpr int iterA = BM * BK / NUM_THREADS;
    constexpr int nrowsA = NUM_THREADS / BK;
    const int idxAk = tid % BK;
    const int idxAm = tid / BK;

    #pragma unroll
    for (int i = 0; i < iterA; i++) {
        const int m = idxAm + i * nrowsA;
        cuda::memcpy_async(
            &As[swizzle<SWZ_B, 0, SWZ_S>(idxAk * BM + m)],
            &A[m * K + idxAk],
            sizeof(float), pipe);
    }
}

// -----------------------------------------------------------------------------
// gmem → smem (async, 16B): vectorized cp.async for A, K-contiguous As[m*BK+k].
// Uses Swizzle<B,2,S> (M=2) to preserve float4 alignment while reducing bank
// conflicts. Requires K-contiguous smem layout (not M-contiguous).
// SWZ_B = ctz(BK) - 2, SWZ_S = ctz(max(NUM_THREADS / (BK/4), BK)).
// -----------------------------------------------------------------------------
template <int BM, int BK, int NUM_THREADS, int SWZ_B, int SWZ_S>
__device__ __forceinline__ void load_A_gmem_to_smem_vec(
    const float* __restrict__ A, int K, float* As, int tid,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    constexpr int VEC = 4;
    constexpr int BK_VEC = BK / VEC;
    constexpr int iterA = BM * BK / NUM_THREADS / VEC;
    constexpr int nrowsA = NUM_THREADS / BK_VEC;
    static_assert(iterA > 0, "at least 1 vectorized loading for A.");

    const int idxAk = (tid % BK_VEC) * VEC;
    const int idxAm = tid / BK_VEC;

    #pragma unroll
    for (int i = 0; i < iterA; i++) {
        const int m = idxAm + i * nrowsA;
        auto* dst_vec = reinterpret_cast<float4*>(&As[swizzle<SWZ_B-2, 2, SWZ_S>(m * BK + idxAk)]);
        const auto* src_vec = reinterpret_cast<const float4*>(&A[m * K + idxAk]);
        cuda::memcpy_async(
            dst_vec, src_vec,
            cuda::aligned_size_t<alignof(float4)>(sizeof(float4)),
            pipe);
    }
}

// -----------------------------------------------------------------------------
// gmem → smem (async, 16B): vectorized cp.async for B, column-major Bs[k+n*BK].
// -----------------------------------------------------------------------------
template <int BN, int BK, int NUM_THREADS>
__device__ __forceinline__ void load_B_gmem_to_smem(
    const float* __restrict__ B, int K, float* Bs, int tid,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    constexpr int VEC = 4;
    constexpr int BK_VEC = BK / VEC;
    constexpr int iterB = BN * BK / NUM_THREADS / VEC;
    constexpr int ncolsB = NUM_THREADS / BK_VEC;
    static_assert(iterB > 0, "at least 1 vectorized loading.");

    const int idxBk = (tid % BK_VEC) * VEC;
    const int idxBn = tid / BK_VEC;

    #pragma unroll
    for (int j = 0; j < iterB; j++) {
        const int n = idxBn + j * ncolsB;
        auto* dst_vec = reinterpret_cast<float4*>(&Bs[n * BK + idxBk]);
        const auto* src_vec = reinterpret_cast<const float4*>(&B[n * K + idxBk]);
        cuda::memcpy_async(
            dst_vec,
            src_vec,
            cuda::aligned_size_t<alignof(float4)>(sizeof(float4)),
            pipe);
    }
}

// -----------------------------------------------------------------------------
// smem → reg: gather this thread's M-slice of As for a fixed k into regM.
// -----------------------------------------------------------------------------
template <int BM, int BK, int WM, int WMITER, int TM, int NTM,
          int SWZ_B, int SWZ_S>
__device__ __forceinline__ void load_A_smem_to_reg(
    const float* As, int k, int warpIdm, int thrIdm, float* regM)
{
    constexpr int WSUBM = WM / WMITER;
    const int warpOffsetM = warpIdm * WM;

    #pragma unroll
    for (int wmi = 0; wmi < WMITER; wmi++) {
        const int offset_m = warpOffsetM + wmi * WSUBM;
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            const int m = offset_m + thrIdm + i * NTM;
            regM[i + wmi * TM] = As[swizzle<SWZ_B, 0, SWZ_S>(k * BM + m)];
        }
    }
}

// -----------------------------------------------------------------------------
// smem → reg: K-contiguous As[m*BK+k] with Swizzle<B,2,S>, matching vec g2s.
// -----------------------------------------------------------------------------
template <int BM, int BK, int WM, int WMITER, int TM, int NTM,
          int SWZ_B, int SWZ_S>
__device__ __forceinline__ void load_A_smem_to_reg_vec(
    const float* As, int k, int warpIdm, int thrIdm, float* regM)
{
    constexpr int WSUBM = WM / WMITER;
    const int warpOffsetM = warpIdm * WM;

    #pragma unroll
    for (int wmi = 0; wmi < WMITER; wmi++) {
        const int offset_m = warpOffsetM + wmi * WSUBM;
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            const int m = offset_m + thrIdm + i * NTM;
            regM[i + wmi * TM] = As[swizzle<SWZ_B-2, 2, SWZ_S>(m * BK + k)];
        }
    }
}

// -----------------------------------------------------------------------------
// smem → reg: KC layout with one precomputed base register per warp-iter.
//
// Key properties:
//   1. swizzle(m*BK + k) = swizzle(m*BK) + k   for k < BK  (async-copy compat)
//   2. swizzle(m_j*BK) = swizzle(m_0*BK) + swizzle(j*NTM*BK)  for all m_0
//      because the XOR bits only depend on the high bits which follow a fixed
//      pattern independent of the per-thread base.
//      => delta[i] = swizzle(i*NTM*BK) is a COMPILE-TIME constant.
//
// So each load is As[kc_base[wmi] + compile_time_delta + compile_time_k]:
//   kc_base[wmi]  — one register per WMITER (vs TM*WMITER=8 without this)
//   compile_time_delta = swizzle(i*NTM*BK) — folded into instruction immediate
//   compile_time_k     — folded into instruction immediate (unrolled loop)
//
// This matches CUTLASS's R226+constant pattern with a single base per section.
//
// Implementation: std::integer_sequence makes i a pack of compile-time
// constants so swizzle<...>(I*NTM*BK) is evaluated at compile time.
// -----------------------------------------------------------------------------
template <int TM, int WMITER, int NTM, int BK, int SWZ_B, int SWZ_S,
          int... Is>
__device__ __forceinline__ void load_A_kc_wmi(
    const float* As, int base, int k, float* regM, int wmi,
    std::integer_sequence<int, Is...>)
{
    // Fold expression: each Is is a compile-time constant
    ((regM[Is + wmi * TM] = As[base + swizzle<SWZ_B-2, 2, SWZ_S>(Is * NTM * BK) + k]), ...);
}

template <int TM, int WMITER, int NTM, int BK, int SWZ_B, int SWZ_S>
__device__ __forceinline__ void load_A_smem_kc_precomp(
    const float* As, const int* kc_base, int k, float* regM)
{
    #pragma unroll
    for (int wmi = 0; wmi < WMITER; wmi++) {
        load_A_kc_wmi<TM, WMITER, NTM, BK, SWZ_B, SWZ_S>(
            As, kc_base[wmi], k, regM, wmi,
            std::make_integer_sequence<int, TM>{});
    }
}

// -----------------------------------------------------------------------------
// smem → reg: gather this thread's N-slice of Bs for a fixed k into regN.
// -----------------------------------------------------------------------------
template <int BK, int WN, int WNITER, int TN, int NTN>
__device__ __forceinline__ void load_B_smem_to_reg(
    const float* Bs, int k, int warpIdn, int thrIdn, float* regN)
{
    constexpr int WSUBN = WN / WNITER;
    const int warpOffsetN = warpIdn * WN;

    #pragma unroll
    for (int wni = 0; wni < WNITER; wni++) {
        const int offset_n = warpOffsetN + wni * WSUBN;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int n = offset_n + thrIdn + j * NTN;
            regN[j + wni * TN] = Bs[k + n * BK];
        }
    }
}

// -----------------------------------------------------------------------------
// Register-level outer product: accum += regM ⊗ regN.
// -----------------------------------------------------------------------------
template <int WMITER, int WNITER, int TM, int TN>
__device__ __forceinline__ void mma_outer_product(
    const float* regM, const float* regN, float* accum)
{
    #pragma unroll
    for (int wni = 0; wni < WNITER; wni++) {
        #pragma unroll
        for (int wmi = 0; wmi < WMITER; wmi++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const int n = j + wni * TN;
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    const int m = i + wmi * TM;
                    accum[m + n * TM * WMITER] += regM[m] * regN[n];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Epilogue: C = alpha * accum + beta * C.
// -----------------------------------------------------------------------------
template <int WM, int WN, int WMITER, int WNITER, int TM, int TN,
          int NTM, int NTN>
__device__ __forceinline__ void epilogue_store(
    float* C, int M, float alpha, float beta, const float* accum,
    int bm, int bn, int warpIdm, int warpIdn, int thrIdm, int thrIdn)
{
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    C += bm + warpIdm * WM + thrIdm + (bn + warpIdn * WN + thrIdn) * M;

    #pragma unroll
    for (int wni = 0; wni < WNITER; wni++) {
        #pragma unroll
        for (int wmi = 0; wmi < WMITER; wmi++) {
            float* Cptr = C + wmi * WSUBM + wni * WSUBN * M;
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    const int accum_m = i + wmi * TM;
                    const int accum_n = j + wni * TN;
                    const int off = i * NTM + j * NTN * M;
                    Cptr[off] = alpha * accum[accum_m + accum_n * TM * WMITER]
                              + beta * Cptr[off];
                }
            }
        }
    }
}

} // namespace db

// -----------------------------------------------------------------------------
// Main kernel. Pipeline schedule (1 in-flight stage at a time, compute runs
// concurrently with the next tile's cp.async):
//
//   prologue : issue tile 0 → commit → wait<0> → sync
//   for each mainloop iter:
//       issue tile (i+1) → commit                 // fire-and-forget
//       compute on tile i                           // overlaps with cp.async
//       wait<0> → sync                              // tile (i+1) landed
//   epilogue : compute on final tile, write C
// -----------------------------------------------------------------------------
// MC=false: K-contiguous As[m*BK+k], Swizzle<B-2,2,S>, vectorized 128-bit g2s
// MC=true:  M-contiguous As[k*BM+m], Swizzle<B,0,S>,   scalar 32-bit g2s
template <int BM = 128, int BN = 128, int BK = 8,
    int WM = 64, int WN = 64,
    int WMITER = 2, int WNITER = 2,
    int TM = 8, int TN = 4,
    bool MC = false>
__global__
__launch_bounds__((BM / WM) * (BN / WN) * 32)
void sgemm_double_buffering_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    assert(is_pow2(M) && is_pow2(N) && is_pow2(K));

    // derived block/warp/thread layout
    constexpr int NWM = BM / WM;
    constexpr int NWN = BN / WN;
    constexpr int NUM_THREADS = NWM * NWN * 32;
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int NTM = WSUBM / TM;
    constexpr int NTN = WSUBN / TN;

    // swizzle params — layout-dependent
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int THR_M = NTM * NWM;
    constexpr int ATOM_M = (THR_M >= BK) ? THR_M : BK;
    // MC=false (KC): Swizzle<B-2, 2, ctz(ATOM_M)> on As[m*BK+k]
    // MC=true:       Swizzle<B,   0, ctz(BM)>     on As[k*BM+m]
    constexpr int SWZ_S = MC ? __builtin_ctz(BM) : __builtin_ctz(ATOM_M);

    // block origin in gmem
    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    A += bm * K;
    B += bn * K;

    __shared__ alignas(128) float As[2][BM * BK];
    __shared__ alignas(128) float Bs[2][BK * BN];

    // warp & lane ids
    const int warpId = threadIdx.x >> 5;
    const int warpIdm = warpId % NWM;
    const int warpIdn = warpId / NWM;
    const int thrId = threadIdx.x & 31;
    const int thrIdm = thrId % NTM;
    const int thrIdn = thrId / NTM;

    float accum[TM * TN * WMITER * WNITER] = {0.0};
    float regM[TM * WMITER];
    float regN[TN * WNITER];
    auto pipe = cuda::make_pipeline();

    // prologue: issue stage 0 and wait for it
    if constexpr (!MC) {
        db::load_A_gmem_to_smem_vec<BM, BK, NUM_THREADS, SWZ_B, SWZ_S>(
            A, K, As[0], threadIdx.x, pipe);
    } else {
        db::load_A_gmem_to_smem<BM, BK, NUM_THREADS, SWZ_B, SWZ_S>(
            A, K, As[0], threadIdx.x, pipe);
    }
    db::load_B_gmem_to_smem<BN, BK, NUM_THREADS>(
        B, K, Bs[0], threadIdx.x, pipe);
    pipe.producer_commit();
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    int read = 0, write = 1;

    // mainloop over K (all but final tile)
    #pragma nounroll
    for (int bk = BK; bk < K; bk += BK) {
        A += BK;
        B += BK;

        // issue next stage — cp.async runs in parallel with the compute below
        if constexpr (!MC) {
            db::load_A_gmem_to_smem_vec<BM, BK, NUM_THREADS, SWZ_B, SWZ_S>(
                A, K, As[write], threadIdx.x, pipe);
        } else {
            db::load_A_gmem_to_smem<BM, BK, NUM_THREADS, SWZ_B, SWZ_S>(
                A, K, As[write], threadIdx.x, pipe);
        }
        db::load_B_gmem_to_smem<BN, BK, NUM_THREADS>(
            B, K, Bs[write], threadIdx.x, pipe);
        pipe.producer_commit();

        // compute on current (read) stage
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            if constexpr (!MC) {
                db::load_A_smem_to_reg_vec<BM, BK, WM, WMITER, TM, NTM, SWZ_B, SWZ_S>(
                    As[read], k, warpIdm, thrIdm, regM);
            } else {
                db::load_A_smem_to_reg<BM, BK, WM, WMITER, TM, NTM, SWZ_B, SWZ_S>(
                    As[read], k, warpIdm, thrIdm, regM);
            }
            db::load_B_smem_to_reg<BK, WN, WNITER, TN, NTN>(
                Bs[read], k, warpIdn, thrIdn, regN);
            db::mma_outer_product<WMITER, WNITER, TM, TN>(regM, regN, accum);
        }

        // ensure the next stage has landed (also acts as the barrier before
        // the next iter's issue can overwrite the now-finished read buffer)
        cuda::pipeline_consumer_wait_prior<0>(pipe);
        __syncthreads();

        read ^= 1;
        write ^= 1;
    }

    // epilogue: compute on final tile
    #pragma unroll
    for (int k = 0; k < BK; k++) {
        if constexpr (!MC) {
            db::load_A_smem_to_reg_vec<BM, BK, WM, WMITER, TM, NTM, SWZ_B, SWZ_S>(
                As[read], k, warpIdm, thrIdm, regM);
        } else {
            db::load_A_smem_to_reg<BM, BK, WM, WMITER, TM, NTM, SWZ_B, SWZ_S>(
                As[read], k, warpIdm, thrIdm, regM);
        }
        db::load_B_smem_to_reg<BK, WN, WNITER, TN, NTN>(
            Bs[read], k, warpIdn, thrIdn, regN);
        db::mma_outer_product<WMITER, WNITER, TM, TN>(regM, regN, accum);
    }

    // write C
    db::epilogue_store<WM, WN, WMITER, WNITER, TM, TN, NTM, NTN>(
        C, M, alpha, beta, accum, bm, bn, warpIdm, warpIdn, thrIdm, thrIdn);
}

template <int BM = 128, int BN = 128, int BK = 8,
    int WM = 64, int WN = 64,
    int WMITER = 2, int WNITER = 2,
    int TM = 8, int TN = 4,
    bool MC = false>
void sgemm_double_buffering(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block((BM / WM) * (BN / WN) * 32);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_double_buffering_kernel<
        BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, MC>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
