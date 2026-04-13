#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * WARPTILING SGEMM (TN): C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * BM, BN, BK: tile size for M, N, K dimension
 * WM, WN: warp tile size for M, N, dimension
 * WMITER, WNITER: number of subtiles for each warp on M, N dimension
 * TM, TN: tile size for thread local register
 *
 * Layouts (best pair, asymmetric):
 *   - A: M-contiguous As[k*BM + m]   (scatter store from float4, unit-stride read per thread)
 *   - B: K-contiguous Bs[k + n*BK]   (single float4 store, broadcast read per warp)
 *
 * # of gmem load: MNK * (1/BM + 1/BN)
 * # of smem load: MNK * (1/TM + 1/TN)
 */

namespace wtv {

// -----------------------------------------------------------------------------
// gmem → smem (vectorized, 128-bit float4): load BM x BK tile of A.
// Reads one float4 per iteration from gmem and scatters the 4 scalars into
// M-contiguous swizzled smem slots.
// -----------------------------------------------------------------------------
template <int BM, int BK, int NUM_THREADS, int SWZ_B, int SWZ_S>
__device__ __forceinline__ void load_A_gmem_to_smem_vec(
    const float* __restrict__ A, int K, float* As, int tid)
{
    constexpr int VEC = 4;
    constexpr int BK_VEC = BK / VEC;
    constexpr int iterA = BM * BK / NUM_THREADS / VEC;
    constexpr int nrowsA = NUM_THREADS / BK_VEC;
    static_assert(iterA > 0, "at least 1 vectorized loading.");

    const int idxAk = (tid % BK_VEC) * VEC;
    const int idxAm = tid / BK_VEC;

    #pragma unroll
    for (int i = 0; i < iterA; i++) {
        const int m = idxAm + i * nrowsA;
        float4 tmp = reinterpret_cast<const float4*>(&A[m * K + idxAk])[0];
        As[swizzle<SWZ_B, 0, SWZ_S>((idxAk + 0) * BM + m)] = tmp.x;
        As[swizzle<SWZ_B, 0, SWZ_S>((idxAk + 1) * BM + m)] = tmp.y;
        As[swizzle<SWZ_B, 0, SWZ_S>((idxAk + 2) * BM + m)] = tmp.z;
        As[swizzle<SWZ_B, 0, SWZ_S>((idxAk + 3) * BM + m)] = tmp.w;
    }
}

// -----------------------------------------------------------------------------
// gmem → smem (vectorized, 128-bit float4): load BK x BN tile of B.
// Both gmem and smem are K-contiguous along the BK axis, so a single float4
// store suffices (no scatter).
// -----------------------------------------------------------------------------
template <int BN, int BK, int NUM_THREADS>
__device__ __forceinline__ void load_B_gmem_to_smem_vec(
    const float* __restrict__ B, int K, float* Bs, int tid)
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
        reinterpret_cast<float4*>(&Bs[n * BK + idxBk])[0] =
            reinterpret_cast<const float4*>(&B[n * K + idxBk])[0];
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

} // namespace wtv

// -----------------------------------------------------------------------------
// Main kernel — composes the helpers in wtv.
// Same algorithm as sgemm_warptiling_kernel, but the gmem→smem stage uses
// 128-bit (float4) loads for better global memory throughput.
// -----------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 8,
    int WM = 64, int WN = 64,
    int WMITER = 2, int WNITER = 2,
    int TM = 8, int TN = 4>
__global__ void sgemm_warptiling_vectorize_kernel(
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

    // swizzle params for As (M-contiguous As[k*BM+m]):
    // thrIdm at bits[0..ctz(NTM)), k at bits[ctz(BM)...) → XOR m with k
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = __builtin_ctz(BM);

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
        wtv::load_A_gmem_to_smem_vec<BM, BK, NUM_THREADS, SWZ_B, SWZ_S>(
            A, K, As, threadIdx.x);
        wtv::load_B_gmem_to_smem_vec<BN, BK, NUM_THREADS>(
            B, K, Bs, threadIdx.x);
        __syncthreads();

        A += BK;
        B += BK;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            wtv::load_A_smem_to_reg<BM, BK, WM, WMITER, TM, NTM, SWZ_B, SWZ_S>(
                As, k, warpIdm, thrIdm, regM);
            wtv::load_B_smem_to_reg<BK, WN, WNITER, TN, NTN>(
                Bs, k, warpIdn, thrIdn, regN);
            wtv::mma_outer_product<WMITER, WNITER, TM, TN>(regM, regN, accum);
        }
        __syncthreads();
    }

    wtv::epilogue_store<WM, WN, WMITER, WNITER, TM, TN, NTM, NTN>(
        C, M, alpha, beta, accum, bm, bn, warpIdm, warpIdn, thrIdm, thrIdn);
}

template <int BM = 128, int BN = 128, int BK = 8,
    int WM = 64, int WN = 64,
    int WMITER = 2, int WNITER = 2,
    int TM = 8, int TN = 4>
void sgemm_warptiling_vectorize(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block((BM / WM) * (BN / WN) * 32);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_warptiling_vectorize_kernel<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
