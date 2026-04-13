#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * WARPTILING SGEMM (TN) with scalar gmem→smem copy: C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * BM, BN, BK: tile size for M, N, K dimension
 * WM, WN: warp tile size for M, N, dimension
 * WMITER, WNITER: number of subtiles for each warp on M, N dimension
 * TM, TN: tile size for thread local register
 *
 * MC=false (default): K-contiguous As[m*BK+k], conflict-free writes, swizzled reads
 * MC=true:            M-contiguous As[k*BM+m], swizzled writes, conflict-free reads
 *
 * Warp tiling with scalar (non-vectorized) gmem→smem loads.
 * This avoids the register overhead of float4 scatter-stores
 * and may improve occupancy.
 */

namespace wt {

// -----------------------------------------------------------------------------
// gmem → smem: load one BM x BK tile of A (row-major, A^T semantics).
// Smem layout is M-contiguous: As[k*BM + m].
// A swizzle is applied to eliminate smem bank conflicts.
// -----------------------------------------------------------------------------
template <int BM, int BK, int NUM_THREADS, int SWZ_B, int SWZ_S, bool MC = false>
__device__ __forceinline__ void load_A_gmem_to_smem(
    const float* __restrict__ A, int K, float* As, int tid)
{
    constexpr int iterA = BM * BK / NUM_THREADS;
    constexpr int nrowsA = NUM_THREADS / BK;
    const int idxAk = tid % BK;
    const int idxAm = tid / BK;

    #pragma unroll
    for (int i = 0; i < iterA; i++) {
        const int m = idxAm + i * nrowsA;
        if constexpr (MC)
            As[swizzle<SWZ_B, 0, SWZ_S>(idxAk * BM + m)] = A[m * K + idxAk];
        else
            As[swizzle<SWZ_B, 0, SWZ_S>(m * BK + idxAk)] = A[m * K + idxAk];
    }
}

// -----------------------------------------------------------------------------
// gmem → smem: load one BK x BN tile of B into column-major Bs[k + n*BK].
// -----------------------------------------------------------------------------
template <int BN, int BK, int NUM_THREADS>
__device__ __forceinline__ void load_B_gmem_to_smem(
    const float* __restrict__ B, int K, float* Bs, int tid)
{
    constexpr int iterB = BK * BN / NUM_THREADS;
    constexpr int ncolsB = NUM_THREADS / BK;
    const int idxBk = tid % BK;
    const int idxBn = tid / BK;

    #pragma unroll
    for (int j = 0; j < iterB; j++) {
        const int n = idxBn + j * ncolsB;
        Bs[idxBk + n * BK] = B[idxBk + n * K];
    }
}

// -----------------------------------------------------------------------------
// smem → reg: gather this thread's M-slice of As for a fixed k into regM.
// Produces TM * WMITER values covering all WMITER sub-tiles owned by the warp.
// -----------------------------------------------------------------------------
template <int BM, int BK, int WM, int WMITER, int TM, int NTM,
          int SWZ_B, int SWZ_S, bool MC = false>
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
            if constexpr (MC)
                regM[i + wmi * TM] = As[swizzle<SWZ_B, 0, SWZ_S>(k * BM + m)];
            else
                regM[i + wmi * TM] = As[swizzle<SWZ_B, 0, SWZ_S>(m * BK + k)];
        }
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
// accum is laid out as accum[m + n * (TM*WMITER)].
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
// Epilogue: C = alpha * accum + beta * C, scattered to each thread's slice.
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

} // namespace wt

// -----------------------------------------------------------------------------
// Main kernel — thin driver that composes the helpers above.
// Algorithm:
//   1. For each K-tile:
//        a. Cooperatively stream A, B tiles from gmem into smem.
//        b. Sync.
//        c. For each k in [0, BK): load regM, regN from smem, FMA into accum.
//        d. Sync.
//   2. Write accum back to C with alpha/beta scaling.
// -----------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 16,
    int WM = 64, int WN = 64,
    int WMITER = 1, int WNITER = 4,
    int TM = 8, int TN = 4, bool MC = false>
__global__ void sgemm_warptiling_kernel(
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

    // MC=true: As[k*BM+m], SWZ_S=ctz(BM) → XOR m-bits with k-bits (fix write conflicts)
    // MC=false: As[m*BK+k], SWZ_S=ctz(BK) → XOR k-bits with m-bits (fix read conflicts)
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = MC ? __builtin_ctz(BM) : __builtin_ctz(BK);

    // block origin in gmem
    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;
    A += bm * K;
    B += bn * K;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

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

    // mainloop over K
    for (int bk = 0; bk < K; bk += BK) {
        wt::load_A_gmem_to_smem<BM, BK, NUM_THREADS, SWZ_B, SWZ_S, MC>(
            A, K, As, threadIdx.x);
        wt::load_B_gmem_to_smem<BN, BK, NUM_THREADS>(
            B, K, Bs, threadIdx.x);
        __syncthreads();

        A += BK;
        B += BK;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            wt::load_A_smem_to_reg<BM, BK, WM, WMITER, TM, NTM, SWZ_B, SWZ_S, MC>(
                As, k, warpIdm, thrIdm, regM);
            wt::load_B_smem_to_reg<BK, WN, WNITER, TN, NTN>(
                Bs, k, warpIdn, thrIdn, regN);
            wt::mma_outer_product<WMITER, WNITER, TM, TN>(regM, regN, accum);
        }
        __syncthreads();
    }

    wt::epilogue_store<WM, WN, WMITER, WNITER, TM, TN, NTM, NTN>(
        C, M, alpha, beta, accum, bm, bn, warpIdm, warpIdn, thrIdm, thrIdn);
}

template <int BM = 128, int BN = 128, int BK = 16,
    int WM = 64, int WN = 64,
    int WMITER = 1, int WNITER = 4,
    int TM = 8, int TN = 4, bool MC = false>
void sgemm_warptiling(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block((BM / WM) * (BN / WN) * 32);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_warptiling_kernel<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, MC>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
