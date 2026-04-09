#pragma once
#include <cassert>
#include "../share.cuh"

/*
 * VECTORIZE SGEMM (TN): C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * BM, BN, BK: tile size for M, N, K dimension
 * TM, TN: tile size for thread local register
 * MC: false = K-contiguous As[m*BK+k] (default), true = M-contiguous As[k*BM+m]
 *
 * KC (default):
 * - gmem store: consecutive threads write consecutive k → no bank conflict
 * - smem read: threads vary in m (stride TM*BK) → swizzle fixes it
 *
 * MC:
 * - gmem store: consecutive threads vary in k → smem stride BM → bank conflicts on stores
 * - smem read: threads vary in m (stride TM) → different conflict pattern
 *
 * # of gmem load: MNK * (1/BM + 1/BN)
 * # of smem load: MNK * (1/TM + 1/TN)
 */

template <int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 8, bool MC = false>
__global__ void sgemm_vectorize_kernel(
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
    int tx = threadIdx.x % Tm;
    int ty = threadIdx.x / Tm;

    // use 128-bit loading
    constexpr int VEC = 4;
    constexpr int BK_VEC = BK / VEC;

    constexpr int iterA = BK * TM * TN / BN / VEC;
    static_assert(iterA > 0, "at least 1 vectorized loading.");
    
    int idxAk = threadIdx.x % BK_VEC * VEC;
    int idxAm = threadIdx.x / BK_VEC;
    int nrowsA = blockDim.x / BK_VEC;

    constexpr int iterB = BK * TM * TN / BM / VEC;
    static_assert(iterB > 0, "at least 1 vectorized loading.");
    int idxBk = threadIdx.x % BK_VEC * VEC;
    int idxBn = threadIdx.x / BK_VEC;
    int ncolsB = blockDim.x / BK_VEC;

    // shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // swizzle parameters depend on layout
    // KC: As[m*BK+k], reads vary in m with stride TM*BK → swizzle from m-bits
    // MC: As[k*BM+m], reads vary in m with stride TM   → swizzle from k-bits
    constexpr int SWZ_B = MC ? __builtin_ctz(TM) : __builtin_ctz(BK);
    constexpr int SWZ_S = MC ? __builtin_ctz(BM)  : __builtin_ctz(BK * TM);
    auto swzA = [](int idx) { return swizzle<SWZ_B, 0, SWZ_S>(idx); };

    float accum[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
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

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                if constexpr (MC) {
                    regM[i] = As[swzA(k * BM + (tx * TM + i))];
                } else {
                    regM[i] = As[swzA((tx * TM + i) * BK + k)];
                }
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regN[j] = Bs[k + (ty * TN + j) * BK];
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

    C += bm + tx * TM + (bn + ty * TN) * M;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            C[i + j * M] = alpha * accum[i + j * TM] + beta * C[i + j * M];
        }
    }
}

template <int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 8, bool MC = false>
void sgemm_vectorize(int M, int N, int K, float alpha,
    const float* A, const float* B, float beta, float* C)
{
    dim3 block(BM * BN / (TM * TN));
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN, MC><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
