#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * TILING SGEMM (TN): C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * BM, BN, BK: tile size for M, N, K dimension
 * TM, TN: tile size for thread local register
 * MC: false = K-contiguous As[m*BK+k] (default), true = M-contiguous As[k*BM+m]
 *
 * # of gmem load: MNK * (1/BM + 1/BN)
 * # of smem load: MNK * (1/TM + 1/TN)
 */

template <int BM = 32, int BN = 32, int BK = 8, int TM = 8, int TN = 8, bool MC = false>
__global__ void sgemm_tiling_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    assert(is_pow2(M) && is_pow2(N) && is_pow2(K));
    assert(BM % TM == 0 && BN % TN == 0);

    // for gmem access
    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;
    
    // advance gmem pointer
    A += bm * K;
    B += bn * K;

    // for local access
    constexpr int Tm = BM / TM;
    constexpr int Tn = BN / TN;
    int tx = threadIdx.x % Tm;
    int ty = threadIdx.x / Tm;

    // iterations for storing smem 
    constexpr int iterA = BK * TM * TN / BN;
    int idxAk = threadIdx.x % BK;
    int idxAm = threadIdx.x / BK;
    int nrowsA = blockDim.x / BK;

    constexpr int iterB = BK * TM * TN / BM;
    int idxBk = threadIdx.x % BK;
    int idxBn = threadIdx.x / BK;
    int ncolsB = blockDim.x / BK;

    // shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // swizzle parameters depend on layout (stride-1 thread mapping: m = tx + i*Tm)
    // KC: As[m*BK+k], k at bits[0..ctz(BK)), tx at bits[ctz(BK)...) → XOR k with tx
    // MC: As[k*BM+m], tx at bits[0..ctz(Tm)), k at bits[ctz(BM)...) → XOR m with k
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = MC ? __builtin_ctz(BM) : __builtin_ctz(BK);
    auto swzA = [](int idx) { return swizzle<SWZ_B, 0, SWZ_S>(idx); };

    float accum[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        // load smem 
        #pragma unroll
        for (int i = 0; i < iterA; i++) {
            int m = idxAm + i * nrowsA;
            if constexpr (MC) {
                As[swzA(idxAk * BM + m)] = A[m * K + idxAk];
            } else {
                As[swzA(m * BK + idxAk)] = A[m * K + idxAk];
            }
        }

        #pragma unroll
        for (int j = 0; j < iterB; j++) {
            int n = idxBn + j * ncolsB;
            Bs[idxBk + n * BK] = B[idxBk + n * K];
        }
        __syncthreads();

        // advance gmem pointer
        A += BK;
        B += BK;

        // compute
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // load into register from smem 
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                if constexpr (MC) {
                    regM[i] = As[swzA(k * BM + tx + i * Tm)];
                } else {
                    regM[i] = As[swzA((tx + i * Tm) * BK + k)];
                }
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regN[j] = Bs[k + (ty + j * Tn) * BK];
            }

            #pragma unroll
            for (int j = 0; j < TN; j++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    accum[i + j * TM] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    C += bm + tx + (bn + ty) * M;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            C[i * Tm + j * Tn * M] = alpha * accum[i + j * TM] + beta * C[i * Tm + j * Tn * M];
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 8, int TM = 8, int TN = 8, bool MC = false>
void sgemm_tiling(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block(BM * BN / (TM * TN));
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_tiling_kernel<BM, BN, BK, TM, TN, MC><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
