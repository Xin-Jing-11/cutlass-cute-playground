#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention "fa3" — FA3-style ping-pong:
//   producer WG  (1 warp, 32 threads): TMA loads K, V, Q (via ClusterTransactionBarrier).
//   consumer WG_A (4 warps, 128 threads): computes QK^T → stores S to ping-pong sS buffer.
//   consumer WG_B (4 warps, 128 threads): reads S, does softmax + PV, maintains O accumulator.
//
// Br is FIXED at 64 in the template for smem budgeting (ping-pong sS at Br=128, Bc=32, FP32
// costs 2×128×32×4 = 32 KB which alone fills most of smem alongside K/V stages; at Br=64 it's
// 16 KB and the whole kernel fits in ~72 KB).
//
// Ping-pong barriers (mbarrier / ClusterBarrier arrive+wait pairs — one-way semaphore):
//   sS_full_bar[2]:  expected=WGA_THREADS (128). WG_A arrives after QK^T; WG_B waits.
//   sS_empty_bar[2]: expected=WGB_THREADS (128). WG_B arrives after PV; WG_A waits.
//   Phase flips per slot on every full cycle (standard hgemm_tma-style pattern).
//   NamedBarrier id 1: consumer-only sync within WG_B for V transpose.

namespace cute_fa_fa3 {

__device__ __forceinline__ float row4_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}

__device__ __forceinline__ float row4_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

} // namespace cute_fa_fa3


template <int Bc, int D_MODEL, int NUM_STAGES,
          class TmaQ, class TmaK, class TmaV,
          class SQLayout, class SKLayout, class SVrLayout>
__global__ static
__launch_bounds__(9 * 32)
void flash_attention_fa3_device(
    int seq_len, int total_rows,
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    SQLayout sQ_layout, SKLayout sK_layout, SVrLayout sVr_layout,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    constexpr int Br = 64;  // hardcoded — see file header
    static_assert(Bc % 16 == 0);
    static_assert(D_MODEL % 16 == 0);
    static_assert(NUM_STAGES >= 2);

    constexpr int WGA_WARPS = 4, WGB_WARPS = 4;
    constexpr int WGA_THREADS = WGA_WARPS * 32;  // 128
    constexpr int WGB_THREADS = WGB_WARPS * 32;  // 128
    constexpr int PROD_THREADS = 32;
    constexpr int WGA_TID_LO = PROD_THREADS;                // 32
    constexpr int WGA_TID_HI = PROD_THREADS + WGA_THREADS;  // 160
    constexpr int WGB_TID_LO = WGA_TID_HI;                  // 160
    constexpr int WGB_TID_HI = WGB_TID_LO + WGB_THREADS;    // 288

    const int bh       = blockIdx.y;
    const int mask_off = bh * seq_len * seq_len;
    const int i0       = blockIdx.x * Br;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const bool is_producer = (warp_id == 0);
    const bool is_wg_a     = (tid >= WGA_TID_LO && tid < WGA_TID_HI);
    const bool is_wg_b     = (tid >= WGB_TID_LO && tid < WGB_TID_HI);

    const int wg_a_tid = tid - WGA_TID_LO;
    const int wg_b_tid = tid - WGB_TID_LO;
    const int wg_a_warp = wg_a_tid >> 5;
    const int wg_b_warp = wg_b_tid >> 5;

    const int row_hi_w = lane >> 2;
    const int row_lo_w = row_hi_w + 8;
    // Global row indices (within CTA's Br block) for WG_A and WG_B. Both cover
    // the SAME 64 rows — only their pipeline stage differs.
    const int row_hi_b_a = wg_a_warp * 16 + row_hi_w;
    const int row_lo_b_a = wg_a_warp * 16 + row_lo_w;
    const int row_hi_b_b = wg_b_warp * 16 + row_hi_w;
    const int row_lo_b_b = wg_b_warp * 16 + row_lo_w;

    const float scale = 1.0f / sqrtf((float)D_MODEL);
    constexpr int MMA_N_QK = Bc / 8;
    constexpr int MMA_N_PV = D_MODEL / 8;

    Tensor mO = make_tensor(make_gmem_ptr(out + bh * seq_len * D_MODEL),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));

    // -------- Shared memory --------
    extern __shared__ __align__(128) char smem_raw[];
    cute::half_t* smem_half = reinterpret_cast<cute::half_t*>(smem_raw);

    constexpr int sQ_size  = cosize_v<decltype(sQ_layout)>;
    constexpr int sK_size  = cosize_v<decltype(sK_layout)>;
    constexpr int sVr_size = cosize_v<decltype(sVr_layout)>;
    constexpr int sV_size  = D_MODEL * Bc;

    cute::half_t* sQ_p  = smem_half;
    cute::half_t* sK_p  = sQ_p  + sQ_size;
    cute::half_t* sVr_p = sK_p  + sK_size;
    cute::half_t* sV_p  = sVr_p + sVr_size;

    constexpr int smem_halves = sQ_size + sK_size + sVr_size + sV_size;

    // sS (FP32): 2 × Br × Bc ping-pong. Placed after half-typed tiles.
    constexpr int sS_off = (smem_halves * 2 + 15) & ~15;
    float* sS_p = reinterpret_cast<float*>(smem_raw + sS_off);
    constexpr int sS_elems_per_buf = Br * Bc;
    auto sS_ptr = [&](int slot) { return sS_p + slot * sS_elems_per_buf; };

    constexpr int sS_total_bytes = 2 * sS_elems_per_buf * sizeof(float);
    constexpr int smem_bar_off   = (sS_off + sS_total_bytes + 15) & ~15;
    uint64_t* q_bar        = reinterpret_cast<uint64_t*>(smem_raw + smem_bar_off);
    uint64_t* kv_bar       = q_bar + 1;
    uint64_t* empty_bar    = kv_bar + NUM_STAGES;
    uint64_t* sS_full_bar  = empty_bar + NUM_STAGES;    // WG_A → WG_B signal
    uint64_t* sS_empty_bar = sS_full_bar + 2;            // WG_B → WG_A signal

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_p),  sQ_layout);
    Tensor sK  = make_tensor(make_smem_ptr(sK_p),  sK_layout);
    Tensor sVr = make_tensor(make_smem_ptr(sVr_p), sVr_layout);

    auto sV_layout_impl = make_layout(make_shape(Int<D_MODEL>{}, Int<Bc>{}),
                                      make_stride(Int<Bc>{}, _1{}));
    Tensor sV = make_tensor(make_smem_ptr(sV_p), sV_layout_impl);

    auto sS_layout = make_layout(make_shape(Int<Br>{}, Int<Bc>{}),
                                 make_stride(Int<Bc>{}, _1{}));

    // Dummy shape references for MMA fragment derivation.
    auto sS_dummy_layout = make_layout(make_shape(Int<Br>{}, Int<Bc>{}),
                                       make_stride(Int<Bc>{}, _1{}));
    auto sO_dummy_layout = make_layout(make_shape(Int<Br>{}, Int<D_MODEL>{}),
                                       make_stride(Int<D_MODEL>{}, _1{}));
    Tensor sS_dummy = make_tensor(make_smem_ptr(sQ_p), sS_dummy_layout);
    Tensor sO_dummy = make_tensor(make_smem_ptr(sQ_p), sO_dummy_layout);

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    // Init barriers (tid 0)
    if (tid == 0) {
        FullBarrier::init(q_bar, 1);
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            FullBarrier::init(&kv_bar[s], 1);
            // Both WG_A and WG_B arrive on empty_bar (each thread once).
            EmptyBarrier::init(&empty_bar[s], WGA_THREADS + WGB_THREADS);
        }
        CUTE_UNROLL
        for (int s = 0; s < 2; ++s) {
            EmptyBarrier::init(&sS_full_bar[s],  WGA_THREADS); // WG_A's 128 arrivals fill it
            EmptyBarrier::init(&sS_empty_bar[s], WGB_THREADS); // WG_B's 128 arrivals drain it
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    const int q_tiles_per_bh  = seq_len / Br;
    const int q_tile_idx      = bh * q_tiles_per_bh + blockIdx.x;
    const int kv_tiles_per_bh = seq_len / Bc;
    const int kv_tile_base    = bh * kv_tiles_per_bh;
    const int T_c             = kv_tiles_per_bh;

    constexpr int q_tx_bytes  = Br * D_MODEL * sizeof(cute::half_t);
    constexpr int kv_tx_bytes = 2 * Bc * D_MODEL * sizeof(cute::half_t);

    // ============================================================
    // PRODUCER (warp 0) — same as wsp
    // ============================================================
    if (is_producer) {
        bool is_leader = (lane == 0);

        Tensor gQ_all   = tma_q.get_tma_tensor(make_shape(total_rows, Int<D_MODEL>{}));
        Tensor gQ_tiled = local_tile(gQ_all, make_shape(Int<Br>{}, Int<D_MODEL>{}),
                                     make_coord(_, 0));
        Tensor gQ_tile  = gQ_tiled(_, _, q_tile_idx);
        auto [tQgQ, tQsQ] = tma_partition(
            tma_q, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(sQ),
            group_modes<0, 2>(gQ_tile));

        if (is_leader) {
            FullBarrier::arrive_and_expect_tx(q_bar, q_tx_bytes);
            copy(tma_q.with(*q_bar), tQgQ, tQsQ);
        }

        Tensor gK = tma_k.get_tma_tensor(make_shape(total_rows, Int<D_MODEL>{}));
        Tensor gV = tma_v.get_tma_tensor(make_shape(total_rows, Int<D_MODEL>{}));
        Tensor gK_tiled = local_tile(gK, make_shape(Int<Bc>{}, Int<D_MODEL>{}), make_coord(_, 0));
        Tensor gV_tiled = local_tile(gV, make_shape(Int<Bc>{}, Int<D_MODEL>{}), make_coord(_, 0));

        auto [tKgK, tKsK] = tma_partition(
            tma_k, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(sK),
            group_modes<0, 2>(gK_tiled));
        auto [tVgV, tVsVr] = tma_partition(
            tma_v, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(sVr),
            group_modes<0, 2>(gV_tiled));

        if (is_leader) {
            CUTE_UNROLL
            for (int s = 0; s < NUM_STAGES; ++s) {
                if (s < T_c) {
                    FullBarrier::arrive_and_expect_tx(&kv_bar[s], kv_tx_bytes);
                    copy(tma_k.with(kv_bar[s]), tKgK(_, kv_tile_base + s), tKsK(_, s));
                    copy(tma_v.with(kv_bar[s]), tVgV(_, kv_tile_base + s), tVsVr(_, s));
                }
            }
        }

        int write_pipe = 0, write_phase = 0;
        for (int tile = NUM_STAGES; tile < T_c; ++tile) {
            EmptyBarrier::wait(&empty_bar[write_pipe], write_phase);
            if (is_leader) {
                FullBarrier::arrive_and_expect_tx(&kv_bar[write_pipe], kv_tx_bytes);
                copy(tma_k.with(kv_bar[write_pipe]), tKgK(_, kv_tile_base + tile), tKsK(_, write_pipe));
                copy(tma_v.with(kv_bar[write_pipe]), tVgV(_, kv_tile_base + tile), tVsVr(_, write_pipe));
            }
            ++write_pipe;
            if (write_pipe == NUM_STAGES) { write_pipe = 0; write_phase ^= 1; }
        }
        return;
    }

    // Wait for Q (both WGs).
    FullBarrier::wait(q_bar, 0);

    // ============================================================
    // CONSUMER WG_A: QK^T → store S to sS[ping_pong_slot]
    // ============================================================
    if (is_wg_a) {
        auto tiled_mma_a = make_tiled_mma(
            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
            Layout<Shape<Int<WGA_WARPS>, _1, _1>>{});
        auto thr_mma_a = tiled_mma_a.get_slice(wg_a_tid);

        Tensor tSrA = thr_mma_a.partition_fragment_A(sQ);
        Tensor tSrB = thr_mma_a.partition_fragment_B(sK(_, _, 0));
        Tensor tSrC = thr_mma_a.partition_fragment_C(sS_dummy);

        auto s2r_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma_a);
        auto s2r_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma_a);
        auto s2r_thr_Q = s2r_copy_Q.get_slice(wg_a_tid);
        auto s2r_thr_K = s2r_copy_K.get_slice(wg_a_tid);

        Tensor tXsQ = s2r_thr_Q.partition_S(sQ);
        Tensor tXrQ = s2r_thr_Q.retile_D(tSrA);
        Tensor tXrK = s2r_thr_K.retile_D(tSrB);

        // Per-tile body: wait kv, compute QK^T, store sS[slot], arrive sS_full, arrive empty.
        auto process_tile = [&](int tile, int slot, int& read_pipe, int& read_phase) {
            FullBarrier::wait(&kv_bar[read_pipe], read_phase);
            Tensor sK_cur = sK(_, _, read_pipe);
            Tensor tXsK_cur = s2r_thr_K.partition_S(sK_cur);

            clear(tSrC);
            #pragma unroll
            for (int k = 0; k < size<2>(tSrA); ++k) {
                copy(s2r_copy_Q, tXsQ(_, _, k), tXrQ(_, _, k));
                copy(s2r_copy_K, tXsK_cur(_, _, k), tXrK(_, _, k));
                gemm(tiled_mma_a, tSrA(_, _, k), tSrB(_, _, k), tSrC);
            }

            Tensor sS_cur = make_tensor(make_smem_ptr(sS_ptr(slot)), sS_layout);
            Tensor tSsS = thr_mma_a.partition_C(sS_cur);
            #pragma unroll
            for (int n = 0; n < MMA_N_QK; ++n) {
                tSsS(0, 0, n) = tSrC(0, 0, n);
                tSsS(1, 0, n) = tSrC(1, 0, n);
                tSsS(2, 0, n) = tSrC(2, 0, n);
                tSsS(3, 0, n) = tSrC(3, 0, n);
            }
            EmptyBarrier::arrive(&sS_full_bar[slot]);
            EmptyBarrier::arrive(&empty_bar[read_pipe]);

            ++read_pipe;
            if (read_pipe == NUM_STAGES) { read_pipe = 0; read_phase ^= 1; }
        };

        int read_pipe = 0, read_phase = 0;

        // Prologue: tiles 0, 1 — no sS_empty wait (slots start empty).
        for (int tile = 0; tile < 2 && tile < T_c; ++tile) {
            process_tile(tile, tile & 1, read_pipe, read_phase);
        }

        // Mainloop: tile >= 2 — wait on sS_empty before reusing a slot.
        int slot = 0, ss_empty_phase = 0;
        for (int tile = 2; tile < T_c; ++tile) {
            EmptyBarrier::wait(&sS_empty_bar[slot], ss_empty_phase);
            process_tile(tile, slot, read_pipe, read_phase);
            slot ^= 1;
            if (slot == 0) ss_empty_phase ^= 1;
        }
        return;
    }

    // ============================================================
    // CONSUMER WG_B: wait on sS, do softmax + PV, maintain O accumulator
    // ============================================================
    if (is_wg_b) {
        auto tiled_mma_b = make_tiled_mma(
            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
            Layout<Shape<Int<WGB_WARPS>, _1, _1>>{});
        auto thr_mma_b = tiled_mma_b.get_slice(wg_b_tid);

        Tensor tOrA = thr_mma_b.partition_fragment_A(sS_dummy);
        Tensor tOrB = thr_mma_b.partition_fragment_B(sV);
        Tensor tOrC = thr_mma_b.partition_fragment_C(sO_dummy);
        clear(tOrC);

        // S-fragment layout for WG_B (to read sS).
        Tensor tSrC = thr_mma_b.partition_fragment_C(sS_dummy);

        auto s2r_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma_b);
        auto s2r_thr_V = s2r_copy_V.get_slice(wg_b_tid);
        Tensor tXsV = s2r_thr_V.partition_S(sV);
        Tensor tXrV = s2r_thr_V.retile_D(tOrB);

        float m_i[2] = { -INFINITY, -INFINITY };
        float l_i[2] = {  0.0f,      0.0f     };

        int read_pipe = 0, read_phase = 0;
        int slot = 0, ss_full_phase = 0;

        for (int tile = 0; tile < T_c; ++tile) {
            // Wait for K+V in smem (for sV-transpose and PV MMA).
            FullBarrier::wait(&kv_bar[read_pipe], read_phase);

            // Transpose sVr[read_pipe] → sV (consumer-local).
            Tensor sVr_cur = sVr(_, _, read_pipe);
            #pragma unroll
            for (int idx = wg_b_tid; idx < Bc * D_MODEL; idx += WGB_THREADS) {
                int r = idx / D_MODEL;
                int c = idx % D_MODEL;
                sV(c, r) = sVr_cur(r, c);
            }
            cutlass::arch::NamedBarrier(WGB_THREADS, 1).sync();

            // Wait for WG_A to fill sS[slot].
            EmptyBarrier::wait(&sS_full_bar[slot], ss_full_phase);

            // Load sS[slot] into tSrC.
            Tensor sS_cur = make_tensor(make_smem_ptr(sS_ptr(slot)), sS_layout);
            Tensor tSsS = thr_mma_b.partition_C(sS_cur);
            #pragma unroll
            for (int n = 0; n < MMA_N_QK; ++n) {
                tSrC(0, 0, n) = tSsS(0, 0, n);
                tSrC(1, 0, n) = tSsS(1, 0, n);
                tSrC(2, 0, n) = tSsS(2, 0, n);
                tSrC(3, 0, n) = tSsS(3, 0, n);
            }
            // Signal WG_A that sS[slot] is free.
            EmptyBarrier::arrive(&sS_empty_bar[slot]);

            // ---- Softmax ----
            float local_max[2] = { -INFINITY, -INFINITY };
            const int j0 = tile * Bc;
            #pragma unroll
            for (int n = 0; n < MMA_N_QK; ++n) {
                int col_base = n * 8 + 2 * (lane & 3);
                int j_a = j0 + col_base;
                int j_b = j0 + col_base + 1;
                float s0 = tSrC(0, 0, n) * scale;
                float s1 = tSrC(1, 0, n) * scale;
                float s2 = tSrC(2, 0, n) * scale;
                float s3 = tSrC(3, 0, n) * scale;
                if (mask) {
                    int r_hi_g = i0 + row_hi_b_b;
                    int r_lo_g = i0 + row_lo_b_b;
                    if (r_hi_g < seq_len && j_a < seq_len) s0 += float(mask[mask_off + r_hi_g*seq_len + j_a]);
                    if (r_hi_g < seq_len && j_b < seq_len) s1 += float(mask[mask_off + r_hi_g*seq_len + j_b]);
                    if (r_lo_g < seq_len && j_a < seq_len) s2 += float(mask[mask_off + r_lo_g*seq_len + j_a]);
                    if (r_lo_g < seq_len && j_b < seq_len) s3 += float(mask[mask_off + r_lo_g*seq_len + j_b]);
                }
                if (j_a >= seq_len) { s0 = -INFINITY; s2 = -INFINITY; }
                if (j_b >= seq_len) { s1 = -INFINITY; s3 = -INFINITY; }
                tSrC(0, 0, n) = s0; tSrC(1, 0, n) = s1;
                tSrC(2, 0, n) = s2; tSrC(3, 0, n) = s3;
                local_max[0] = fmaxf(local_max[0], fmaxf(s0, s1));
                local_max[1] = fmaxf(local_max[1], fmaxf(s2, s3));
            }
            float m_ij0 = cute_fa_fa3::row4_reduce_max(local_max[0]);
            float m_ij1 = cute_fa_fa3::row4_reduce_max(local_max[1]);

            float local_sum[2] = { 0.0f, 0.0f };
            #pragma unroll
            for (int n = 0; n < MMA_N_QK; ++n) {
                float e0 = isfinite(m_ij0) ? __expf(tSrC(0, 0, n) - m_ij0) : 0.0f;
                float e1 = isfinite(m_ij0) ? __expf(tSrC(1, 0, n) - m_ij0) : 0.0f;
                float e2 = isfinite(m_ij1) ? __expf(tSrC(2, 0, n) - m_ij1) : 0.0f;
                float e3 = isfinite(m_ij1) ? __expf(tSrC(3, 0, n) - m_ij1) : 0.0f;
                tSrC(0, 0, n) = e0; tSrC(1, 0, n) = e1;
                tSrC(2, 0, n) = e2; tSrC(3, 0, n) = e3;
                local_sum[0] += e0 + e1;
                local_sum[1] += e2 + e3;
            }
            float l_ij0 = cute_fa_fa3::row4_reduce_sum(local_sum[0]);
            float l_ij1 = cute_fa_fa3::row4_reduce_sum(local_sum[1]);

            float m_i_new0 = fmaxf(m_i[0], m_ij0);
            float m_i_new1 = fmaxf(m_i[1], m_ij1);
            float alpha0 = isfinite(m_i[0]) ? __expf(m_i[0] - m_i_new0) : 0.0f;
            float alpha1 = isfinite(m_i[1]) ? __expf(m_i[1] - m_i_new1) : 0.0f;
            float beta0  = isfinite(m_ij0)  ? __expf(m_ij0  - m_i_new0) : 0.0f;
            float beta1  = isfinite(m_ij1)  ? __expf(m_ij1  - m_i_new1) : 0.0f;
            l_i[0] = alpha0 * l_i[0] + beta0 * l_ij0;
            l_i[1] = alpha1 * l_i[1] + beta1 * l_ij1;
            m_i[0] = m_i_new0;
            m_i[1] = m_i_new1;

            #pragma unroll
            for (int n = 0; n < MMA_N_PV; ++n) {
                tOrC(0, 0, n) *= alpha0; tOrC(1, 0, n) *= alpha0;
                tOrC(2, 0, n) *= alpha1; tOrC(3, 0, n) *= alpha1;
            }
            #pragma unroll
            for (int n = 0; n < MMA_N_QK; ++n) {
                tSrC(0, 0, n) *= beta0; tSrC(1, 0, n) *= beta0;
                tSrC(2, 0, n) *= beta1; tSrC(3, 0, n) *= beta1;
            }

            constexpr int MMA_K_PV = Bc / 16;
            #pragma unroll
            for (int k_pv = 0; k_pv < MMA_K_PV; ++k_pv) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    tOrA(i,     0, k_pv) = cute::half_t(tSrC(i, 0, 2 * k_pv    ));
                    tOrA(i + 4, 0, k_pv) = cute::half_t(tSrC(i, 0, 2 * k_pv + 1));
                }
            }

            #pragma unroll
            for (int k = 0; k < size<2>(tOrA); ++k) {
                copy(s2r_copy_V, tXsV(_, _, k), tXrV(_, _, k));
                gemm(tiled_mma_b, tOrA(_, _, k), tOrB(_, _, k), tOrC);
            }

            EmptyBarrier::arrive(&empty_bar[read_pipe]);
            cutlass::arch::NamedBarrier(WGB_THREADS, 1).sync();

            slot ^= 1;
            if (slot == 0) ss_full_phase ^= 1;
            ++read_pipe;
            if (read_pipe == NUM_STAGES) { read_pipe = 0; read_phase ^= 1; }
        }

        // Epilogue (WG_B writes O).
        float inv_l0 = (l_i[0] > 0.0f) ? 1.0f / l_i[0] : 0.0f;
        float inv_l1 = (l_i[1] > 0.0f) ? 1.0f / l_i[1] : 0.0f;
        const int r_hi_g = i0 + row_hi_b_b;
        const int r_lo_g = i0 + row_lo_b_b;
        #pragma unroll
        for (int n = 0; n < MMA_N_PV; ++n) {
            int col_base = n * 8 + 2 * (lane & 3);
            if (r_hi_g < seq_len) {
                mO(r_hi_g, col_base    ) = cute::half_t(tOrC(0, 0, n) * inv_l0);
                mO(r_hi_g, col_base + 1) = cute::half_t(tOrC(1, 0, n) * inv_l0);
            }
            if (r_lo_g < seq_len) {
                mO(r_lo_g, col_base    ) = cute::half_t(tOrC(2, 0, n) * inv_l1);
                mO(r_lo_g, col_base + 1) = cute::half_t(tOrC(3, 0, n) * inv_l1);
            }
        }
    }
}


template <int Bc = 32, int D_MODEL = 128, int NUM_STAGES = 2>
void flash_attention_fa3(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    assert(d_model == D_MODEL);
    constexpr int Br = 64;
    assert(seq_len % Br == 0 && seq_len % Bc == 0);

    auto sQ_layout  = tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::half_t>{},
                                    make_shape(Int<Br>{}, Int<D_MODEL>{}));
    auto sK_layout  = tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::half_t>{},
                                    make_shape(Int<Bc>{}, Int<D_MODEL>{}, Int<NUM_STAGES>{}));
    auto sVr_layout = tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::half_t>{},
                                    make_shape(Int<Bc>{}, Int<D_MODEL>{}, Int<NUM_STAGES>{}));

    const int total_rows = batch_size * num_heads * seq_len;
    Tensor mQ_desc = make_tensor(Q, make_shape(total_rows, Int<D_MODEL>{}),
                                 make_stride(Int<D_MODEL>{}, _1{}));
    Tensor mK_desc = make_tensor(K, make_shape(total_rows, Int<D_MODEL>{}),
                                 make_stride(Int<D_MODEL>{}, _1{}));
    Tensor mV_desc = make_tensor(V, make_shape(total_rows, Int<D_MODEL>{}),
                                 make_stride(Int<D_MODEL>{}, _1{}));

    auto tma_q = make_tma_atom(SM90_TMA_LOAD{}, mQ_desc, sQ_layout,
                               make_shape(Int<Br>{}, Int<D_MODEL>{}));
    auto tma_k = make_tma_atom(SM90_TMA_LOAD{}, mK_desc, sK_layout(_, _, 0),
                               make_shape(Int<Bc>{}, Int<D_MODEL>{}));
    auto tma_v = make_tma_atom(SM90_TMA_LOAD{}, mV_desc, sVr_layout(_, _, 0),
                               make_shape(Int<Bc>{}, Int<D_MODEL>{}));

    constexpr int sQ_size  = cosize_v<decltype(sQ_layout)>;
    constexpr int sK_size  = cosize_v<decltype(sK_layout)>;
    constexpr int sVr_size = cosize_v<decltype(sVr_layout)>;
    constexpr int sV_size  = D_MODEL * Bc;
    constexpr int smem_halves = sQ_size + sK_size + sVr_size + sV_size;
    constexpr int sS_off = (smem_halves * 2 + 15) & ~15;
    constexpr int sS_bytes = 2 * Br * Bc * sizeof(float);
    constexpr int smem_bar_off   = (sS_off + sS_bytes + 15) & ~15;
    // q_bar(1) + kv_bar(NS) + empty_bar(NS) + sS_full(2) + sS_empty(2)
    constexpr int smem_bar_bytes = (1 + 2 * NUM_STAGES + 4) * sizeof(uint64_t);
    constexpr int smem_bytes     = smem_bar_off + smem_bar_bytes;

    auto kernel = flash_attention_fa3_device<
        Bc, D_MODEL, NUM_STAGES,
        decltype(tma_q), decltype(tma_k), decltype(tma_v),
        decltype(sQ_layout), decltype(sK_layout), decltype(sVr_layout)>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(9 * 32);  // 1 producer + 4 WG_A + 4 WG_B warps
    kernel<<<grid, block, smem_bytes>>>(
        seq_len, total_rows, tma_q, tma_k, tma_v,
        sQ_layout, sK_layout, sVr_layout,
        mask, out);
}
