#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cassert>

/*
 * Multistage tensor-core HGEMM via nvcuda::wmma + __pipeline_memcpy_async.
 * C = alpha * A^T * B + beta * C   (TN col-major).
 *
 *   A(K,M) col-major, ldA = K
 *   B(K,N) col-major, ldB = K
 *   C(M,N) col-major, ldC = M
 *
 * NUM_STAGES circular smem buffers:
 *   prologue  : issue NUM_STAGES async loads and commit each as its own stage
 *   mainloop  : wait for the oldest pending stage, compute on it, issue the
 *               next stage into the same slot.  wmma accumulator fragments
 *               persist across the K-loop.
 */

namespace cuda_hgemm_multistage {

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NUM_STAGES>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32)
void hgemm_multistage_device(
    int M, int N, int K,
    float alpha,
    const half* __restrict__ A, int ldA,
    const half* __restrict__ B, int ldB,
    float beta,
    half* __restrict__ C, int ldC)
{
    using namespace nvcuda;
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int NUM_WARPS    = WARPS_M * WARPS_N;
    constexpr int NUM_THREADS  = NUM_WARPS * 32;
    constexpr int M_WARP_TILES = BM / (WARPS_M * WM);
    constexpr int N_WARP_TILES = BN / (WARPS_N * WN);
    constexpr int K_BLOCKS     = BK / WK;
    constexpr int PAD          = 8;

    static_assert(BM % (WARPS_M * WM) == 0);
    static_assert(BN % (WARPS_N * WN) == 0);
    static_assert(BK % WK == 0);
    static_assert(NUM_STAGES >= 2);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int wm_idx  = warp_id / WARPS_N;
    const int wn_idx  = warp_id % WARPS_N;

    const int cta_m = blockIdx.x * BM;
    const int cta_n = blockIdx.y * BN;

    const int warp_m_base = wm_idx * M_WARP_TILES * WM;
    const int warp_n_base = wn_idx * N_WARP_TILES * WN;

    __shared__ __align__(16) half sA[NUM_STAGES][BM][BK + PAD];
    __shared__ __align__(16) half sB[NUM_STAGES][BN][BK + PAD];
    __shared__ __align__(16) float epi[NUM_WARPS][WM][WN];

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[M_WARP_TILES][N_WARP_TILES];
    #pragma unroll
    for (int mi = 0; mi < M_WARP_TILES; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_WARP_TILES; ++ni)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    constexpr int VEC    = 8;
    constexpr int BK_VEC = BK / VEC;
    static_assert(BK % VEC == 0);
    static_assert(NUM_THREADS % BK_VEC == 0);
    constexpr int ROWS_PER_IT = NUM_THREADS / BK_VEC;

    const int row_in_vec  = tid / BK_VEC;
    const int col_vec_idx = tid % BK_VEC;
    const int col_base    = col_vec_idx * VEC;

    // Async gmem -> smem for one stage (does NOT commit). BM rows of A, BN of B.
    auto issue_stage = [&](int s, int k_tile) {
        #pragma unroll
        for (int rb = 0; rb < BM; rb += ROWS_PER_IT) {
            const int m_loc = rb + row_in_vec;
            const int m_gl  = cta_m + m_loc;
            const int k_gl  = k_tile + col_base;
            if (m_gl < M && k_gl + VEC <= K) {
                __pipeline_memcpy_async(
                    &sA[s][m_loc][col_base],
                    &A[k_gl + m_gl * ldA],
                    16);
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    sA[s][m_loc][col_base + v] =
                        (m_gl < M && (k_gl + v) < K)
                            ? A[(k_gl + v) + m_gl * ldA]
                            : __float2half(0.0f);
                }
            }
        }
        #pragma unroll
        for (int rb = 0; rb < BN; rb += ROWS_PER_IT) {
            const int n_loc = rb + row_in_vec;
            const int n_gl  = cta_n + n_loc;
            const int k_gl  = k_tile + col_base;
            if (n_gl < N && k_gl + VEC <= K) {
                __pipeline_memcpy_async(
                    &sB[s][n_loc][col_base],
                    &B[k_gl + n_gl * ldB],
                    16);
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    sB[s][n_loc][col_base + v] =
                        (n_gl < N && (k_gl + v) < K)
                            ? B[(k_gl + v) + n_gl * ldB]
                            : __float2half(0.0f);
                }
            }
        }
    };

    auto compute_stage = [&](int s) {
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[M_WARP_TILES];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag[N_WARP_TILES];
            #pragma unroll
            for (int mi = 0; mi < M_WARP_TILES; ++mi) {
                wmma::load_matrix_sync(
                    a_frag[mi],
                    &sA[s][warp_m_base + mi * WM][kb * WK],
                    BK + PAD);
            }
            #pragma unroll
            for (int ni = 0; ni < N_WARP_TILES; ++ni) {
                wmma::load_matrix_sync(
                    b_frag[ni],
                    &sB[s][warp_n_base + ni * WN][kb * WK],
                    BK + PAD);
            }
            #pragma unroll
            for (int mi = 0; mi < M_WARP_TILES; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < N_WARP_TILES; ++ni) {
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }
    };

    const int K_TILES = (K + BK - 1) / BK;

    // --- Prologue: issue up to NUM_STAGES async loads ------------------------
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s) {
        if (s < K_TILES) {
            issue_stage(s, s * BK);
        }
        __pipeline_commit();
    }

    // --- Mainloop: wait oldest, compute, issue next --------------------------
    for (int k_tile_idx = 0; k_tile_idx < K_TILES; ++k_tile_idx) {
        const int s = k_tile_idx % NUM_STAGES;

        __pipeline_wait_prior(NUM_STAGES - 1);
        __syncthreads();

        compute_stage(s);

        __syncthreads();

        const int next = k_tile_idx + NUM_STAGES;
        if (next < K_TILES) {
            issue_stage(s, next * BK);
        }
        __pipeline_commit();
    }

    // --- Epilogue: alpha*acc + beta*C → gmem  (per-warp 16x16 tile) ---------
    #pragma unroll
    for (int mi = 0; mi < M_WARP_TILES; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_WARP_TILES; ++ni) {
            wmma::store_matrix_sync(
                &epi[warp_id][0][0],
                c_frag[mi][ni], WN, wmma::mem_row_major);
            __syncwarp();

            const int m_row = warp_m_base + mi * WM;
            const int n_col = warp_n_base + ni * WN;

            #pragma unroll
            for (int e = 0; e < (WM * WN) / 32; ++e) {
                const int idx = lane_id * ((WM * WN) / 32) + e;
                const int r = idx / WN;
                const int c = idx % WN;
                const int m_gl = cta_m + m_row + r;
                const int n_gl = cta_n + n_col + c;
                if (m_gl < M && n_gl < N) {
                    const float acc  = epi[warp_id][r][c];
                    const float cold = __half2float(C[m_gl + n_gl * ldC]);
                    const float out  = alpha * acc + beta * cold;
                    C[m_gl + n_gl * ldC] = __float2half(out);
                }
            }
            __syncwarp();
        }
    }
}

// -----------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32,
          int WARPS_M = 2, int WARPS_N = 2, int NUM_STAGES = 3>
void hgemm_multistage(
    int M, int N, int K,
    float alpha,
    const half* A, int ldA,
    const half* B, int ldB,
    float beta,
    half* C, int ldC)
{
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
    dim3 block(NUM_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    hgemm_multistage_device<BM, BN, BK, WARPS_M, WARPS_N, NUM_STAGES>
        <<<grid, block>>>(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

}  // namespace cuda_hgemm_multistage
