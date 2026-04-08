#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * TILING SGEMM (TN): C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 * 
 * BM, BN, BK: tile size for M, N, K dimension
 * TM, TN: tile size for thread local register
 * 
 * # of gmem load: MNK * (1/BM + 1/BN)
 * # of smem load: MNK * (1/TM + 1/TN)
 */

template <int BM = 32, int BN = 32, int BK = 8, int TM = 8, int TN = 8>
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
    A += bm * K; // (bm, 0)
    B += bn * K; // (0, bn)

    // for local access 
    constexpr int Tm = BM / TM; // # of thread in M dimension
    // int Tn = BN / TN; // # of thread in N dimension

    int tx = threadIdx.x % Tm; // [0, BM/TM)
    int ty = threadIdx.x / Tm; // [0, BN/TN)

    // for loading shared memory
    constexpr int iterA = BK * TM * TN / BN; // number of iterations loaded by each thread 
    int idxAm = threadIdx.x / BK;
    int idxAk = threadIdx.x % BK;
    int nrowsA = blockDim.x / BK;

    constexpr int iterB = BK * TM * TN / BM; // number of iterations loaded by each thread 
    int idxBk = threadIdx.x % BK;
    int idxBn = threadIdx.x / BK;
    int ncolsB = blockDim.x / BK;

    // shared memory 
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // swizzle functions to reduce bank conflict
    // S must match where the thread-varying bits are in each smem's linear index
    // As[m * BK + k]: threads vary in m with stride TM → varying bit at log2(BK*TM)
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = __builtin_ctz(BK * TM);
    auto swzA = [](int idx) { return swizzle<SWZ_B, 0, SWZ_S>(idx); };

    // accumulator in register 
    float accum[TM * TN] = {0.0f};
    // register cache for tile of A and B
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        // load shared memory
        #pragma unroll
        for (int i = 0; i < iterA; i++) {
            int m = idxAm + i * nrowsA;
            As[swzA(m * BK + idxAk)] = A[m * K + idxAk];
        }

        #pragma unroll
        for (int j = 0; j < iterB; j++) {
            int n = idxBn + j * ncolsB;
            Bs[idxBk + n * BK] = B[idxBk + n * K];
        }
        __syncthreads();

        // advance gmem pointer
        A += BK; // (bm, bk)
        B += BK; // (bk, bn)

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // load tile into register
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regM[i] = As[swzA((tx * TM + i) * BK + k)];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regN[j] = Bs[k + (ty * TN + j) * BK];
            }

            // compute
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

    // epilogue: write back to gmem
    C += bm + tx * TM + (bn + ty * TN) * M; // (bm + tx*TM, bn + ty*TN)
    for (int j = 0; j < TN; j++) {
        for (int i = 0; i < TM; i++) {
            C[i + j * M] = alpha * accum[i + j * TM] + beta * C[i + j * M];
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 8, int TM = 8, int TN = 8>
void sgemm_tiling(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block(BM * BN / (TM * TN));
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_tiling_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
