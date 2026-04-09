#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * SMEM SGEMM (TN): C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * MC: false = K-contiguous As[k + m*BS] (default), true = M-contiguous As[m + k*BS]
 *
 * # of gmem load: 2MNK/BLOCK_SIZE
 * # of smem load: 2MNK
 */

template <int BLOCK_SIZE = 32, bool MC = false>
__global__ void sgemm_smem_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    assert(is_pow2(M) && is_pow2(N) && is_pow2(K));

    // for gmem access
    int bm = blockIdx.x * BLOCK_SIZE;
    int bn = blockIdx.y * BLOCK_SIZE;

    // advance gmem pointer
    A += bm * K;    // (bm, 0)
    B += bn * K;    // (0, bn)

    // for local access
    int tx = threadIdx.x % BLOCK_SIZE;
    int ty = threadIdx.x / BLOCK_SIZE;

    // shared memory
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // swizzle function to reduce bank conflict
    // KC: As[k + m*BS], reads vary in k (stride 1) with m fixed per iteration → swizzle on m-bits
    // MC: As[m + k*BS], reads vary in m (stride 1) with k fixed per iteration → swizzle on k-bits
    constexpr int SWZ_BITS = __builtin_ctz(BLOCK_SIZE);
    auto swz = [](int idx) { return swizzle<SWZ_BITS, 0, SWZ_BITS>(idx); };

    float acc = 0.0f;
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        // load smem
        if constexpr (MC) {
            // As(m, k) stored as As[m + k*BS] — M contiguous
            As[swz(ty + tx * BLOCK_SIZE)] = A[ty * K + tx];
        } else {
            // As(k, m) stored as As[k + m*BS] — K contiguous
            As[swz(tx + ty * BLOCK_SIZE)] = A[ty * K + tx];
        }
        // Bs(k, n) = B(bk + k, bn + n)
        Bs[tx + ty * BLOCK_SIZE] = B[tx + ty * K];
        __syncthreads();

        // advance gmem pointer
        A += BLOCK_SIZE;    // (bm, bk)
        B += BLOCK_SIZE;    // (bk, bn)

        // As access has bank conflict if without swizzle. Bs broadcast.
        #pragma unroll
        for (auto i = 0; i < BLOCK_SIZE; i++) {
            if constexpr (MC) {
                // read As[tx + i*BS] — consecutive tx are stride 1 (M contiguous)
                acc += As[swz(tx + i * BLOCK_SIZE)] * Bs[i + ty * BLOCK_SIZE];
            } else {
                // read As[i + tx*BS] — varying tx at stride BS
                acc += As[swz(i + tx * BLOCK_SIZE)] * Bs[i + ty * BLOCK_SIZE];
            }
        }
        __syncthreads();
    }

    C[bm + tx + (bn + ty) * M] = alpha * acc + beta * C[bm + tx + (bn + ty) * M];
}

template <int BLOCK_SIZE = 32, bool MC = false>
void sgemm_smem(
    int M, int N, int K,
    float alpha,
    const float* A, const float* B,
    float beta,
    float* C)
{
    dim3 block(BLOCK_SIZE * BLOCK_SIZE);
    dim3 grid(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    sgemm_smem_kernel<BLOCK_SIZE, MC><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
