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
 * MC: false = K-contiguous As[m*BK+k] (default), true = M-contiguous As[k*BM+m]
 *
 * # of gmem load: MNK * (1/BM + 1/BN)
 * # of smem load: MNK * (1/TM + 1/TN)
 */

template <int BM = 128, int BN = 128, int BK = 8,  
    int WM = 64, int WN = 64,
    int WMITER = 2, int WNITER = 2,
    int TM = 8, int TN = 4, bool MC = false>
__global__ void sgemm_warptiling_vectorize_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    assert(is_pow2(M) && is_pow2(N) && is_pow2(K));
    constexpr int NWM = BM / WM; // number of warps in m direction
    constexpr int NWN = BN / WN; // number of warps in n direction
    assert(blockDim.x == NWM * NWN * 32); // number of warp should match threads count

    // for gmem access
    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;
    
    // advance gmem pointer
    A += bm * K;
    B += bn * K;

    // shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // calculate warp (wx, wy) in 2d warps layout
    const int warpId = threadIdx.x >> 5;
    const int warpIdm = warpId % NWM;
    const int warpIdn = warpId / NWM;

    // subtile index 
    constexpr int WSUBM = WM / WMITER; // subtile size m for each warp
    constexpr int WSUBN = WN / WNITER; // subtile size n for each warp
    constexpr int NTM = WSUBM / TM; // number of thread in m direction for each warp
    constexpr int NTN = WSUBN / TN; // number of thread in n direction for each warp
    const int thrId = threadIdx.x & ((1<<5) - 1);
    const int thrIdm = thrId % NTM; 
    const int thrIdn = thrId / NTM; 

    // use 128-bit loading
    constexpr int VEC = 4;
    constexpr int BK_VEC = BK / VEC;

    // swizzle parameters depend on layout (stride-1 thread mapping: m = thrIdm + i*NTM)
    // KC: As[m*BK+k], k at bits[0..ctz(BK)), thrIdm at bits[ctz(BK)...) → XOR k with thrIdm
    // MC: As[k*BM+m], thrIdm at bits[0..ctz(NTM)), k at bits[ctz(BM)...) → XOR m with k
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = MC ? __builtin_ctz(BM) : __builtin_ctz(BK);
    auto swzA = [](int idx) { return swizzle<SWZ_B, 0, SWZ_S>(idx); };

    // for smem storing
    constexpr int iterA = BM * BK / (NWM * NWN * 32) / VEC;
    static_assert(iterA > 0, "at least 1 vectorized loading.");
    int idxAk = threadIdx.x % BK_VEC * VEC;
    int idxAm = threadIdx.x / BK_VEC;
    int nrowsA = blockDim.x / BK_VEC;

    constexpr int iterB = BN * BK / (NWM * NWN * 32) / VEC;
    static_assert(iterB > 0, "at least 1 vectorized loading.");
    int idxBk = threadIdx.x % BK_VEC * VEC;
    int idxBn = threadIdx.x / BK_VEC;
    int ncolsB = blockDim.x / BK_VEC;

    // register 
    float accum[TM * TN * WMITER * WNITER] = {0.0};
    float regM[TM * WMITER] = {0.0};
    float regN[TN * WNITER] = {0.0};
    
    for (int bk = 0; bk < K; bk += BK) {
        // load gmem to smem 
        #pragma unroll
        for (int i = 0; i < iterA; i++) {
            int m = idxAm + i * nrowsA;
            float4 tmp = reinterpret_cast<const float4*>(&A[m * K + idxAk])[0];
            if constexpr (MC) {
                // As[k*BM + m] — M contiguous
                As[swzA((idxAk + 0) * BM + m)] = tmp.x;
                As[swzA((idxAk + 1) * BM + m)] = tmp.y;
                As[swzA((idxAk + 2) * BM + m)] = tmp.z;
                As[swzA((idxAk + 3) * BM + m)] = tmp.w;
            } else {
                // As[m*BK + k] — K contiguous
                As[swzA(m * BK + idxAk + 0)] = tmp.x;
                As[swzA(m * BK + idxAk + 1)] = tmp.y;
                As[swzA(m * BK + idxAk + 2)] = tmp.z;
                As[swzA(m * BK + idxAk + 3)] = tmp.w;
            }
        }

        #pragma unroll
        for (int j = 0; j < iterB; j++) {
            int n = idxBn + j * ncolsB;
            reinterpret_cast<float4*>(&Bs[n * BK + idxBk])[0] = 
                reinterpret_cast<const float4*>(&B[n * K + idxBk])[0];
        }
        __syncthreads();

        // advance gmem pointer
        A += BK;
        B += BK;
        
        // load from smem to register and compute
        for (int k = 0; k < BK; k++) {
            const int warpOffsetM = warpIdm * WM;
            // load regM for all WMITER sub-tiles
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
            // load regN for all WNITER sub-tiles
            for (int wni = 0; wni < WNITER; wni++) {
                const int offset_n = warpOffsetN + wni * WSUBN;
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    const int n = offset_n + thrIdn + j * NTN;
                    regN[j + wni * TN] = Bs[k + n * BK];
                }
            }

            // compute all WMITER x WNITER outer products
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

    // write back accumulation
    // C is col-major (M,N):(1,M). Base pointer to this thread's warp origin.
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

template <int BM = 128, int BN = 128, int BK = 8,  
    int WM = 64, int WN = 64,
    int WMITER = 2, int WNITER = 2,
    int TM = 8, int TN = 4, bool MC = false>
void sgemm_warptiling_vectorize(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block((BM / WM) * (BN / WN) * 32);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_warptiling_vectorize_kernel<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN, MC>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
