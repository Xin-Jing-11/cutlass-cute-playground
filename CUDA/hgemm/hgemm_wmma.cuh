#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cassert>

/*
 * Tensor-core HGEMM via nvcuda::wmma (no inline PTX, no CUTLASS).
 * C = alpha * A^T * B + beta * C   (TN col-major).
 *
 * Storage (matches the CUTLASS HGEMM convention):
 *   A(K,M) col-major, ldA = K
 *   B(K,N) col-major, ldB = K
 *   C(M,N) col-major, ldC = M
 *
 * WMMA uses 16x16x16 fragments: matrix_a row_major, matrix_b col_major,
 * float accumulator. Each warp owns an M_WARP_TILES x N_WARP_TILES slab of
 * 16x16 output tiles; K_BLOCKS = BK / 16 inner steps per CTA K-tile.
 */

namespace cuda_hgemm_wmma {

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32)
void hgemm_wmma_device(
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
    constexpr int PAD          = 8;    // half padding; 16B = 1 sector off bank 0

    static_assert(BM % (WARPS_M * WM) == 0, "BM must tile WARPS_M * 16");
    static_assert(BN % (WARPS_N * WN) == 0, "BN must tile WARPS_N * 16");
    static_assert(BK % WK == 0,              "BK must be multiple of 16");

    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;
    const int wm_idx   = warp_id / WARPS_N;
    const int wn_idx   = warp_id % WARPS_N;

    const int cta_m = blockIdx.x * BM;
    const int cta_n = blockIdx.y * BN;

    // Per-warp output block base (contiguous partition)
    const int warp_m_base = wm_idx * M_WARP_TILES * WM;
    const int warp_n_base = wn_idx * N_WARP_TILES * WN;

    // Shared memory tiles.  sA is row-major [m][k]; sB is col-major [n][k].
    __shared__ __align__(16) half sA[BM][BK + PAD];
    __shared__ __align__(16) half sB[BN][BK + PAD];

    // Epilogue scratch: one 16x16 float tile per warp.
    __shared__ __align__(16) float epi[NUM_WARPS][WM][WN];

    // Accumulators, held across all K tiles.
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[M_WARP_TILES][N_WARP_TILES];
    #pragma unroll
    for (int mi = 0; mi < M_WARP_TILES; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_WARP_TILES; ++ni)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    // 128-bit (8 half) vectorised gmem->smem tile loads.
    constexpr int VEC           = 8;
    constexpr int BK_VEC        = BK / VEC;
    static_assert(BK % VEC == 0, "BK must be multiple of 8 for 128-bit vec load");
    static_assert(NUM_THREADS % BK_VEC == 0,
                  "BK_VEC must divide NUM_THREADS for tiled row distribution");
    constexpr int ROWS_A_PER_IT = NUM_THREADS / BK_VEC;
    constexpr int ROWS_B_PER_IT = NUM_THREADS / BK_VEC;
    static_assert(BM % ROWS_A_PER_IT == 0, "BM / ROWS_A_PER_IT must be integral");
    static_assert(BN % ROWS_B_PER_IT == 0, "BN / ROWS_B_PER_IT must be integral");

    const int row_in_vec  = tid / BK_VEC;     // 0 .. ROWS_*_PER_IT-1
    const int col_vec_idx = tid % BK_VEC;     // 0 .. BK_VEC-1
    const int col_base    = col_vec_idx * VEC;

    for (int k_tile = 0; k_tile < K; k_tile += BK) {

        // --- Load A tile: sA[m][k] = A[k_tile+k, cta_m+m] = A_gmem[k + m*ldA] ---
        #pragma unroll
        for (int rb = 0; rb < BM; rb += ROWS_A_PER_IT) {
            const int m_loc = rb + row_in_vec;
            const int m_gl  = cta_m + m_loc;
            const int k_gl  = k_tile + col_base;
            if (m_gl < M && k_gl + VEC <= K) {
                *reinterpret_cast<uint4*>(&sA[m_loc][col_base]) =
                    *reinterpret_cast<const uint4*>(&A[k_gl + m_gl * ldA]);
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    sA[m_loc][col_base + v] =
                        (m_gl < M && (k_gl + v) < K)
                            ? A[(k_gl + v) + m_gl * ldA]
                            : __float2half(0.0f);
                }
            }
        }

        // --- Load B tile: sB[n][k] = B[k_tile+k, cta_n+n] = B_gmem[k + n*ldB] ---
        #pragma unroll
        for (int rb = 0; rb < BN; rb += ROWS_B_PER_IT) {
            const int n_loc = rb + row_in_vec;
            const int n_gl  = cta_n + n_loc;
            const int k_gl  = k_tile + col_base;
            if (n_gl < N && k_gl + VEC <= K) {
                *reinterpret_cast<uint4*>(&sB[n_loc][col_base]) =
                    *reinterpret_cast<const uint4*>(&B[k_gl + n_gl * ldB]);
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    sB[n_loc][col_base + v] =
                        (n_gl < N && (k_gl + v) < K)
                            ? B[(k_gl + v) + n_gl * ldB]
                            : __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // --- Inner K loop: wmma load + mma ---
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[M_WARP_TILES];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag[N_WARP_TILES];

            #pragma unroll
            for (int mi = 0; mi < M_WARP_TILES; ++mi) {
                const int m_row = warp_m_base + mi * WM;
                wmma::load_matrix_sync(
                    a_frag[mi],
                    &sA[m_row][kb * WK],
                    BK + PAD);
            }
            #pragma unroll
            for (int ni = 0; ni < N_WARP_TILES; ++ni) {
                const int n_col = warp_n_base + ni * WN;
                wmma::load_matrix_sync(
                    b_frag[ni],
                    &sB[n_col][kb * WK],
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
        __syncthreads();
    }

    // --- Epilogue: D = alpha*acc + beta*C  (per warp, one 16x16 tile at a time)
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

            // 32 threads cover 256 elements → 8 per thread.
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
// Host launcher
// -----------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 16,
          int WARPS_M = 2, int WARPS_N = 2>
void hgemm_wmma(
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
    hgemm_wmma_device<BM, BN, BK, WARPS_M, WARPS_N><<<grid, block>>>(
        M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

}  // namespace cuda_hgemm_wmma
