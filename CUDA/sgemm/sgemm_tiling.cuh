#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * Tiling SGEMM: C = alpha * A * B + beta * C
 * A(M,K), B(K,N), C(M,N), all column-major, single precision.
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
    A += bm;     // (bm, 0)
    B += bn * K; // (0, bn)

    // for local access 
    int Tm = BM / TM; // # of thread in M dimension
    // int Tn = BN / TN; // # of thread in N dimension

    int tx = threadIdx.x % Tm; // [0, BM/TM)
    int ty = threadIdx.x / Tm; // [0, BN/TN)

    // for loading shared memory, number of elements loaded by each thread 
    int strideA = BK * TM * TN / BN; // stride for loading A to smem, in element
    int offsetA = strideA * threadIdx.x; // offset for loading A to smem, in element
    int idxAm = offsetA % BM;
    int idxAk = offsetA / BM;

    int strideB = BK * TM * TN / BM; // stride for loading B to smem, in element
    int offsetB = strideB * threadIdx.x; // offset for loading B to smem, in element
    int idxBk = offsetB % BK;
    int idxBn = offsetB / BK;

    // shared memory 
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // accumulator in register 
    float accum[TM * TN] = {0.0f};
    // register cache for tile of A and B
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        // load shared memory 
        for (int i = 0; i < strideA; i++) {
            As[offsetA + i] = A[idxAm + i + idxAk * M];
        }
        
        for (int j = 0; j < strideB; j++) {
            Bs[offsetB + j] = B[idxBk + j + idxBn * K];
        }
        __syncthreads();

        // advance gmem pointer
        A += BK * M; // (bm, bk)
        B += BK;     // (bk, bn)

        for (int k = 0; k < BK; k++) {
            // load tile into register 
            for (int i = 0; i < TM; i++) {
                regM[i] = As[tx * TM + i + k * BM];
            }
            for (int j = 0; j < TN; j++) {
                regN[j] = Bs[k + (ty * TN + j) * BK];
            }

            // compute
            for (int j = 0; j < TN; j++) {
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
