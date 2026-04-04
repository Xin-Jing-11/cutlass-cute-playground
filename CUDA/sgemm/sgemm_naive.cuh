#pragma once
#include "../share.cuh"
/*
 * Naive SGEMM: C = alpha * A * B + beta * C
 * A(M,K), B(K,N), C(M,N), all column-major, single precision.
 * 
 * # of gmem load: 2MNK
 */

template <int BLOCK_SIZE = 32>
__global__ void sgemm_naive_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    int m = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int n = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (m < M && n < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            // broadcast A and coalesce B 
            acc += A[m + k * M] * B[k + n * K];
        }
        C[m + n * M] = alpha * acc + beta * C[m + n * M];
    }
}

template <int BLOCK_SIZE = 32>
void sgemm_naive(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    sgemm_naive_kernel<BLOCK_SIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
