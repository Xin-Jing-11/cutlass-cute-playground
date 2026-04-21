#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <mma.h>
#include <cassert>

/*
 * Warp-specialized HGEMM via TMA + nvcuda::wmma (no inline PTX).
 * C = alpha * A^T * B + beta * C   (TN col-major).
 *
 * CTA = PRODUCER_WARPS + CONSUMER_WARPS warps.
 *   - producer WG (first 32*PRODUCER_WARPS threads): a single elect_one
 *     thread issues cde::cp_async_bulk_tensor_2d_global_to_shared + arrive_tx.
 *   - consumer WG (remaining 32*CONSUMER_WARPS threads): wmma 16x16x16
 *     fragments, 4x2 warp tiling by default, axpby epilogue.
 *
 * Per-stage sync uses cuda::barrier<thread_scope_block>:
 *   full_bar[s] : init count = 1  → only TMA leader arrives (arrive_tx)
 *   empty_bar[s]: init count = NUM_CONSUMER_THREADS → every consumer arrives
 * Phase-parity waits (wait_parity) let consumers wait without arriving and
 * reuse the barrier across K-tiles without re-init.
 *
 * Runs on SM120 when compiled for sm_120a (for bulk-TMA PTX).
 */

namespace cuda_hgemm_tma {

namespace cde = cuda::device::experimental;
using bar_t = cuda::barrier<cuda::thread_scope_block>;

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NUM_STAGES,
          int PRODUCER_WARPS = 4>
__global__ __launch_bounds__((WARPS_M * WARPS_N + PRODUCER_WARPS) * 32)
void hgemm_tma_device(
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    int M, int N, int K,
    float alpha, float beta,
    half* __restrict__ C, int ldC)
{
    using namespace nvcuda;
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int NUM_CONSUMER_WARPS   = WARPS_M * WARPS_N;
    constexpr int NUM_CONSUMER_THREADS = NUM_CONSUMER_WARPS * 32;
    constexpr int NUM_PRODUCER_THREADS = PRODUCER_WARPS * 32;
    constexpr int NUM_THREADS          = NUM_PRODUCER_THREADS + NUM_CONSUMER_THREADS;
    constexpr int M_WARP_TILES         = BM / (WARPS_M * WM);
    constexpr int N_WARP_TILES         = BN / (WARPS_N * WN);
    constexpr int K_BLOCKS             = BK / WK;

    static_assert(BM % (WARPS_M * WM) == 0);
    static_assert(BN % (WARPS_N * WN) == 0);
    static_assert(BK % WK == 0);
    static_assert(NUM_STAGES >= 2);

    const int tid          = threadIdx.x;
    const int warp_id      = tid / 32;
    const int lane_id      = tid % 32;
    const bool is_producer = (tid < NUM_PRODUCER_THREADS);
    const int cons_tid     = tid - NUM_PRODUCER_THREADS;
    const int cons_warp    = cons_tid / 32;
    const int wm_idx       = cons_warp / WARPS_N;
    const int wn_idx       = cons_warp % WARPS_N;

    const int cta_m = blockIdx.x * BM;
    const int cta_n = blockIdx.y * BN;
    const int warp_m_base = wm_idx * M_WARP_TILES * WM;
    const int warp_n_base = wn_idx * N_WARP_TILES * WN;

    // Shared storage — smem tiles + per-stage barriers + epilogue scratch.
    __shared__ __align__(128) half sA[NUM_STAGES][BM][BK];
    __shared__ __align__(128) half sB[NUM_STAGES][BN][BK];
    __shared__ bar_t full_bar[NUM_STAGES];
    __shared__ bar_t empty_bar[NUM_STAGES];
    __shared__ __align__(16) float epi[NUM_CONSUMER_WARPS][WM][WN];

    // Barrier init — one thread, fence before anyone uses them.
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            init(&full_bar[s],  1);                        // only TMA leader arrives
            init(&empty_bar[s], NUM_CONSUMER_THREADS);     // every consumer arrives
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    constexpr uint32_t TMA_TX_BYTES =
        BM * BK * sizeof(half) + BN * BK * sizeof(half);

    const int K_TILES = (K + BK - 1) / BK;

    // -----------------------------------------------------------------------
    // PRODUCER
    // -----------------------------------------------------------------------
    if (is_producer) {
        // Elect one lane in warp 0 of the producer WG to drive TMA.
        const bool tma_leader = (warp_id == 0) && (lane_id == 0);

        // Prologue: issue up to NUM_STAGES TMAs. empty_bar is still at phase 0
        // (consumer hasn't arrived), so we must NOT advance write_phase here.
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            if (s < K_TILES && tma_leader) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sA[s], &tmap_A, s * BK, cta_m, full_bar[s]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sB[s], &tmap_B, s * BK, cta_n, full_bar[s]);
                (void) cuda::device::barrier_arrive_tx(full_bar[s], 1, TMA_TX_BYTES);
            }
        }

        int write_pipe  = 0;
        int write_phase = 0;

        for (int k_tile_idx = NUM_STAGES; k_tile_idx < K_TILES; ++k_tile_idx) {
            // Wait for consumers to release write_pipe (phase alternates 0/1).
            empty_bar[write_pipe].wait_parity(write_phase);
            if (tma_leader) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sA[write_pipe], &tmap_A, k_tile_idx * BK, cta_m, full_bar[write_pipe]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &sB[write_pipe], &tmap_B, k_tile_idx * BK, cta_n, full_bar[write_pipe]);
                (void) cuda::device::barrier_arrive_tx(full_bar[write_pipe], 1, TMA_TX_BYTES);
            }
            ++write_pipe;
            if (write_pipe == NUM_STAGES) { write_pipe = 0; write_phase ^= 1; }
        }
        return;
    }

    // -----------------------------------------------------------------------
    // CONSUMER
    // -----------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[M_WARP_TILES][N_WARP_TILES];
    #pragma unroll
    for (int mi = 0; mi < M_WARP_TILES; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_WARP_TILES; ++ni)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    int read_pipe  = 0;
    int read_phase = 0;

    for (int k_tile_idx = 0; k_tile_idx < K_TILES; ++k_tile_idx) {
        full_bar[read_pipe].wait_parity(read_phase);

        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[M_WARP_TILES];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag[N_WARP_TILES];

            #pragma unroll
            for (int mi = 0; mi < M_WARP_TILES; ++mi) {
                wmma::load_matrix_sync(
                    a_frag[mi],
                    &sA[read_pipe][warp_m_base + mi * WM][kb * WK],
                    BK);
            }
            #pragma unroll
            for (int ni = 0; ni < N_WARP_TILES; ++ni) {
                wmma::load_matrix_sync(
                    b_frag[ni],
                    &sB[read_pipe][warp_n_base + ni * WN][kb * WK],
                    BK);
            }
            #pragma unroll
            for (int mi = 0; mi < M_WARP_TILES; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < N_WARP_TILES; ++ni) {
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }

        // Signal producer that this stage is free.
        (void) empty_bar[read_pipe].arrive();

        ++read_pipe;
        if (read_pipe == NUM_STAGES) { read_pipe = 0; read_phase ^= 1; }
    }

    // --- Epilogue ----------------------------------------------------------
    #pragma unroll
    for (int mi = 0; mi < M_WARP_TILES; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_WARP_TILES; ++ni) {
            wmma::store_matrix_sync(
                &epi[cons_warp][0][0],
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
                    const float acc  = epi[cons_warp][r][c];
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
// Host: build CUtensorMap for a (K,M) or (K,N) col-major half matrix and launch.
// -----------------------------------------------------------------------------
inline CUresult build_tma_descriptor(
    CUtensorMap* tmap,
    const half* data,
    uint64_t outer_dim,        // M (for A) or N (for B)
    uint64_t inner_dim,        // K
    uint32_t box_inner,        // BK
    uint32_t box_outer)        // BM or BN
{
    // 2D descriptor; inner (K) is contiguous in memory, outer (M or N) strides.
    CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    uint64_t size[2]        = { inner_dim, outer_dim };
    uint64_t stride[1]      = { inner_dim * sizeof(half) };  // bytes per outer-dim step
    uint32_t box_size[2]    = { box_inner, box_outer };
    uint32_t elem_stride[2] = { 1, 1 };
    return cuTensorMapEncodeTiled(
        tmap, dtype, 2,
        const_cast<half*>(data), size, stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,              // no swizzle — wmma expects plain layout
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BM = 128, int BN = 128, int BK = 32,
          int WARPS_M = 4, int WARPS_N = 2, int NUM_STAGES = 3,
          int PRODUCER_WARPS = 4>
void hgemm_tma(
    int M, int N, int K,
    float alpha,
    const half* A, int ldA,
    const half* B, int ldB,
    float beta,
    half* C, int ldC)
{
    CUtensorMap tmap_A{}, tmap_B{};
    (void) build_tma_descriptor(&tmap_A, A, /*outer=*/M, /*inner=*/K, BK, BM);
    (void) build_tma_descriptor(&tmap_B, B, /*outer=*/N, /*inner=*/K, BK, BN);

    constexpr int NUM_THREADS = (WARPS_M * WARPS_N + PRODUCER_WARPS) * 32;
    dim3 block(NUM_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    auto kernel = hgemm_tma_device<BM, BN, BK, WARPS_M, WARPS_N, NUM_STAGES, PRODUCER_WARPS>;

    // All smem is declared static inside the kernel — nvcc sizes it automatically.
    // If total exceeds 48KB, opt in to the larger dynamic smem carve-out.
    constexpr int kSmemStatic = NUM_STAGES * (BM * BK + BN * BK) * (int) sizeof(half)
                              + (WARPS_M * WARPS_N) * 16 * 16 * (int) sizeof(float);
    if constexpr (kSmemStatic > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemStatic);
    }

    kernel<<<grid, block>>>(tmap_A, tmap_B, M, N, K, alpha, beta, C, ldC);
}

}  // namespace cuda_hgemm_tma
