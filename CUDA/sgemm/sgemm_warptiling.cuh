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
 * MC: false = K-contiguous As[m*BK+k] (default), true = M-contiguous As[k*BM+m]
 * 
 * Warp tiling with scalar (non-vectorized) gmem→smem loads.
 * This avoids the register overhead of float4 scatter-stores
 * and may improve occupancy.
 */

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
    constexpr int NWM = BM / WM;
    constexpr int NWN = BN / WN;
    constexpr int NUM_THREADS = NWM * NWN * 32;

    // for gmem access
    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;

    A += bm * K;
    B += bn * K;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // warp layout
    const int warpId = threadIdx.x >> 5;
    const int warpIdm = warpId % NWM;
    const int warpIdn = warpId / NWM;

    // thread within warp
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int NTM = WSUBM / TM;
    constexpr int NTN = WSUBN / TN;
    const int thrId = threadIdx.x & 31;
    const int thrIdm = thrId % NTM;
    const int thrIdn = thrId / NTM;

    // swizzle (stride-1 thread mapping: m = thrIdm + i*NTM)
    // KC: As[m*BK+k], k at bits[0..ctz(BK)), thrIdm at bits[ctz(BK)...) → XOR k with thrIdm
    // MC: As[k*BM+m], thrIdm at bits[0..ctz(NTM)), k at bits[ctz(BM)...) → XOR m with k
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = MC ? __builtin_ctz(BM) : __builtin_ctz(BK);
    auto swzA = [](int idx) { return swizzle<SWZ_B, 0, SWZ_S>(idx); };

    // scalar gmem→smem indices
    constexpr int iterA = BM * BK / NUM_THREADS;
    int idxAk = threadIdx.x % BK;
    int idxAm = threadIdx.x / BK;
    int nrowsA = NUM_THREADS / BK;

    constexpr int iterB = BK * BN / NUM_THREADS;
    int idxBk = threadIdx.x % BK;
    int idxBn = threadIdx.x / BK;
    int ncolsB = NUM_THREADS / BK;

    float accum[TM * TN * WMITER * WNITER] = {0.0};
    float regM[TM * WMITER] = {0.0};
    float regN[TN * WNITER] = {0.0};

    for (int bk = 0; bk < K; bk += BK) {
        // scalar gmem→smem for A
        #pragma unroll
        for (int i = 0; i < iterA; i++) {
            int m = idxAm + i * nrowsA;
            if constexpr (MC) {
                As[swzA(idxAk * BM + m)] = A[m * K + idxAk];
            } else {
                As[swzA(m * BK + idxAk)] = A[m * K + idxAk];
            }
        }

        // scalar gmem→smem for B
        #pragma unroll
        for (int j = 0; j < iterB; j++) {
            int n = idxBn + j * ncolsB;
            Bs[idxBk + n * BK] = B[idxBk + n * K];
        }
        __syncthreads();

        A += BK;
        B += BK;

        // compute
        for (int k = 0; k < BK; k++) {
            const int warpOffsetM = warpIdm * WM;
            for (int wmi = 0; wmi < WMITER; wmi++) {
                const int offset_m = warpOffsetM + wmi * WSUBM;
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    const int m = offset_m + thrIdm + i * NTM;
                    if constexpr (MC) {
                        regM[i + wmi * TM] = As[swzA(k * BM + m)];
                    } else {
                        regM[i + wmi * TM] = As[swzA(m * BK + k)];
                    }
                }
            }

            const int warpOffsetN = warpIdn * WN;
            for (int wni = 0; wni < WNITER; wni++) {
                const int offset_n = warpOffsetN + wni * WSUBN;
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    const int n = offset_n + thrIdn + j * NTN;
                    regN[j + wni * TN] = Bs[k + n * BK];
                }
            }

            for (int wni = 0; wni < WNITER; wni++) {
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
        __syncthreads();
    }

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
                    Cptr[i * NTM + j * NTN * M] = alpha * accum[accum_m + accum_n * TM * WMITER]
                                    + beta * Cptr[i * NTM + j * NTN * M];
                }
            }
        }
    }
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
