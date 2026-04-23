#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention "wgmma" — TMA + WGMMA (SM90) warp-specialized.
//
// SM90 warp-specialized design:
//   WG0 (threads 0-127):   Producer — elect_one issues TMA loads
//   WG1 (threads 128-255): Consumer — WGMMA SS for QK^T and PV
//
// Br=64 fixed (WGMMA M=64). V loaded transposed by TMA: gmem view
// (D, total_rows) stride (1, D), into smem sVt(D, Bc) MN-major.
// No explicit thread transpose needed.
// P stored to sP between QK^T and PV for WGMMA A operand.
//
// Requirements:
//   Bc ∈ {64, 128}, D_MODEL = 128, NUM_STAGES >= 2
//   seq_len % 64 == 0, seq_len % Bc == 0

namespace cute_fa_wgmma {

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

} // namespace cute_fa_wgmma


template <int Bc, int D_MODEL, int NUM_STAGES,
          class TmaQ, class TmaK, class TmaVt,
          class SQLayout, class SKLayout, class SVtLayout,
          class SPLayout,
          class MMA_QK, class MMA_PV>
__global__ static
__launch_bounds__(256)
void flash_attention_wgmma_device(
    int seq_len, int total_rows,
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    SQLayout sQ_layout, SKLayout sK_layout, SVtLayout sVt_layout,
    SPLayout sP_layout,
    MMA_QK mma_qk, MMA_PV mma_pv,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    constexpr int Br = 64;
    constexpr int WG_SIZE = 128;
    static_assert(Bc % 16 == 0);
    static_assert(D_MODEL % 16 == 0);
    static_assert(NUM_STAGES >= 2);

    const int wg_idx   = threadIdx.x / WG_SIZE;
    const int bh       = blockIdx.y;
    const int mask_off = bh * seq_len * seq_len;
    const int i0       = blockIdx.x * Br;

    // -------- Shared memory layout --------
    extern __shared__ __align__(128) char smem_raw[];
    cute::half_t* smem_base = reinterpret_cast<cute::half_t*>(smem_raw);

    constexpr int sQ_size  = cosize_v<SQLayout>;
    constexpr int sK_size  = cosize_v<SKLayout>;
    constexpr int sVt_size = cosize_v<SVtLayout>;
    constexpr int sP_sz    = cosize_v<SPLayout>;

    cute::half_t* sQ_p  = smem_base;
    cute::half_t* sK_p  = sQ_p  + sQ_size;
    cute::half_t* sVt_p = sK_p  + sK_size;
    cute::half_t* sP_p  = sVt_p + sVt_size;

    constexpr int smem_halves  = sQ_size + sK_size + sVt_size + sP_sz;
    constexpr int smem_bar_off = (smem_halves * 2 + 127) & ~127;
    uint64_t* q_bar     = reinterpret_cast<uint64_t*>(smem_raw + smem_bar_off);
    uint64_t* kv_bar    = q_bar + 1;
    uint64_t* empty_bar = kv_bar + NUM_STAGES;

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_p),  sQ_layout);
    Tensor sK  = make_tensor(make_smem_ptr(sK_p),  sK_layout);
    Tensor sVt = make_tensor(make_smem_ptr(sVt_p), sVt_layout);
    Tensor sP  = make_tensor(make_smem_ptr(sP_p),  sP_layout);

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    // -------- Init barriers --------
    if (threadIdx.x == 0) {
        FullBarrier::init(q_bar, 1);
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            FullBarrier::init(&kv_bar[s], 1);
            EmptyBarrier::init(&empty_bar[s], 1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    const int q_tiles_per_bh  = seq_len / Br;
    const int q_tile_idx      = bh * q_tiles_per_bh + blockIdx.x;
    const int kv_tiles_per_bh = seq_len / Bc;
    const int kv_tile_base    = bh * kv_tiles_per_bh;
    const int T_c             = kv_tiles_per_bh;

    constexpr int q_tx_bytes  = Br * D_MODEL * int(sizeof(cute::half_t));
    // K: Bc*D bytes, Vt: D*Bc bytes (same total)
    constexpr int kv_tx_bytes = 2 * Bc * D_MODEL * int(sizeof(cute::half_t));

    // ================================================================
    // PRODUCER (WG0, threads 0-127)
    // ================================================================
    if (wg_idx == 0) {
        bool is_leader = (threadIdx.x == 0);

        // Q TMA (once)
        Tensor gQ = tma_q.get_tma_tensor(make_shape(total_rows, Int<D_MODEL>{}));
        Tensor gQ_tiled = local_tile(gQ, make_shape(Int<Br>{}, Int<D_MODEL>{}),
                                     make_coord(_, 0));
        auto [tQgQ, tQsQ] = tma_partition(
            tma_q, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(sQ), group_modes<0, 2>(gQ_tiled));

        if (is_leader) {
            FullBarrier::arrive_and_expect_tx(q_bar, q_tx_bytes);
            copy(tma_q.with(*q_bar), tQgQ(_, q_tile_idx), tQsQ);
        }

        // K TMA
        Tensor gK = tma_k.get_tma_tensor(make_shape(total_rows, Int<D_MODEL>{}));
        Tensor gK_tiled = local_tile(gK, make_shape(Int<Bc>{}, Int<D_MODEL>{}),
                                     make_coord(_, 0));
        auto [tKgK, tKsK] = tma_partition(
            tma_k, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(sK), group_modes<0, 2>(gK_tiled));

        // Vt TMA: transposed gmem view (D, total_rows)
        Tensor gVt = tma_vt.get_tma_tensor(make_shape(Int<D_MODEL>{}, total_rows));
        Tensor gVt_tiled = local_tile(gVt, make_shape(Int<D_MODEL>{}, Int<Bc>{}),
                                      make_coord(0, _));
        auto [tVgVt, tVsVt] = tma_partition(
            tma_vt, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(sVt), group_modes<0, 2>(gVt_tiled));

        // Prologue
        if (is_leader) {
            CUTE_UNROLL
            for (int s = 0; s < NUM_STAGES; ++s) {
                if (s < T_c) {
                    FullBarrier::arrive_and_expect_tx(&kv_bar[s], kv_tx_bytes);
                    copy(tma_k.with(kv_bar[s]),
                         tKgK(_, kv_tile_base + s), tKsK(_, s));
                    copy(tma_vt.with(kv_bar[s]),
                         tVgVt(_, kv_tile_base + s), tVsVt(_, s));
                }
            }
        }

        // Mainloop
        int write_pipe = 0, write_phase = 0;
        for (int tile = NUM_STAGES; tile < T_c; ++tile) {
            EmptyBarrier::wait(&empty_bar[write_pipe], write_phase);
            if (is_leader) {
                FullBarrier::arrive_and_expect_tx(&kv_bar[write_pipe], kv_tx_bytes);
                copy(tma_k.with(kv_bar[write_pipe]),
                     tKgK(_, kv_tile_base + tile), tKsK(_, write_pipe));
                copy(tma_vt.with(kv_bar[write_pipe]),
                     tVgVt(_, kv_tile_base + tile), tVsVt(_, write_pipe));
            }
            ++write_pipe;
            if (write_pipe == NUM_STAGES) { write_pipe = 0; write_phase ^= 1; }
        }
        return;
    }

    // ================================================================
    // CONSUMER (WG1, threads 128-255)
    // ================================================================
    const int consumer_tid = threadIdx.x - WG_SIZE;
    const int warp_in_wg   = consumer_tid / 32;
    const int lane         = consumer_tid & 31;
    const int row_hi_w     = lane >> 2;
    const int row_lo_w     = row_hi_w + 8;
    const int row_hi_b     = warp_in_wg * 16 + row_hi_w;
    const int row_lo_b     = warp_in_wg * 16 + row_lo_w;

    const float scale = 1.0f / sqrtf((float)D_MODEL);
    constexpr int MMA_N_QK = Bc / 8;
    constexpr int MMA_N_PV = D_MODEL / 8;

    Tensor mO = make_tensor(make_gmem_ptr(out + bh * seq_len * D_MODEL),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));

    // Wait for Q TMA
    FullBarrier::wait(q_bar, 0);

    // ---- WGMMA QK^T setup ----
    auto thr_qk = mma_qk.get_slice(consumer_tid);
    Tensor tSsQ = thr_qk.partition_A(sQ);
    Tensor tSsK = thr_qk.partition_B(sK);

    // C fragment for QK^T
    auto tSrC = [&]() {
        auto dS = make_tensor(make_smem_ptr(sQ_p),
                              make_layout(make_shape(Int<Br>{}, Int<Bc>{}),
                                          make_stride(Int<Bc>{}, _1{})));
        return thr_qk.make_fragment_C(thr_qk.partition_C(dS));
    }();

    // ---- WGMMA PV setup ----
    auto thr_pv = mma_pv.get_slice(consumer_tid);
    Tensor tOsP  = thr_pv.partition_A(sP);
    Tensor tOsVt = thr_pv.partition_B(sVt);

    // O accumulator
    auto tOrC = [&]() {
        auto dO = make_tensor(make_smem_ptr(sQ_p),
                              make_layout(make_shape(Int<Br>{}, Int<D_MODEL>{}),
                                          make_stride(Int<D_MODEL>{}, _1{})));
        return thr_pv.make_fragment_C(thr_pv.partition_C(dO));
    }();
    clear(tOrC);

    // Softmax state
    float m_i[2] = { -INFINITY, -INFINITY };
    float l_i[2] = {  0.0f,      0.0f     };

    int read_pipe = 0, read_phase = 0;

    for (int tile = 0; tile < T_c; ++tile) {
        int j0 = tile * Bc;

        // Wait for KV TMA
        FullBarrier::wait(&kv_bar[read_pipe], read_phase);

        // ---- WGMMA QK^T ----
        clear(tSrC);
        warpgroup_fence_operand(tSrC);
        warpgroup_arrive();
        gemm(mma_qk, tSsQ, tSsK(_, _, _, read_pipe), tSrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tSrC);

        // ---- Softmax ----
        float local_max[2] = { -INFINITY, -INFINITY };
        #pragma unroll
        for (int g = 0; g < MMA_N_QK; ++g) {
            int col_base = g * 8 + 2 * (lane & 3);
            int j_a = j0 + col_base;
            int j_b = j0 + col_base + 1;

            float s0 = tSrC(g*4+0) * scale;
            float s1 = tSrC(g*4+1) * scale;
            float s2 = tSrC(g*4+2) * scale;
            float s3 = tSrC(g*4+3) * scale;

            if (mask) {
                int r_hi_g = i0 + row_hi_b;
                int r_lo_g = i0 + row_lo_b;
                if (r_hi_g < seq_len && j_a < seq_len)
                    s0 += float(mask[mask_off + r_hi_g * seq_len + j_a]);
                if (r_hi_g < seq_len && j_b < seq_len)
                    s1 += float(mask[mask_off + r_hi_g * seq_len + j_b]);
                if (r_lo_g < seq_len && j_a < seq_len)
                    s2 += float(mask[mask_off + r_lo_g * seq_len + j_a]);
                if (r_lo_g < seq_len && j_b < seq_len)
                    s3 += float(mask[mask_off + r_lo_g * seq_len + j_b]);
            }
            if (j_a >= seq_len) { s0 = -INFINITY; s2 = -INFINITY; }
            if (j_b >= seq_len) { s1 = -INFINITY; s3 = -INFINITY; }

            tSrC(g*4+0) = s0; tSrC(g*4+1) = s1;
            tSrC(g*4+2) = s2; tSrC(g*4+3) = s3;

            local_max[0] = fmaxf(local_max[0], fmaxf(s0, s1));
            local_max[1] = fmaxf(local_max[1], fmaxf(s2, s3));
        }
        float m_ij0 = cute_fa_wgmma::row4_reduce_max(local_max[0]);
        float m_ij1 = cute_fa_wgmma::row4_reduce_max(local_max[1]);

        float local_sum[2] = { 0.0f, 0.0f };
        #pragma unroll
        for (int g = 0; g < MMA_N_QK; ++g) {
            float e0 = isfinite(m_ij0) ? __expf(tSrC(g*4+0) - m_ij0) : 0.0f;
            float e1 = isfinite(m_ij0) ? __expf(tSrC(g*4+1) - m_ij0) : 0.0f;
            float e2 = isfinite(m_ij1) ? __expf(tSrC(g*4+2) - m_ij1) : 0.0f;
            float e3 = isfinite(m_ij1) ? __expf(tSrC(g*4+3) - m_ij1) : 0.0f;
            tSrC(g*4+0) = e0; tSrC(g*4+1) = e1;
            tSrC(g*4+2) = e2; tSrC(g*4+3) = e3;
            local_sum[0] += e0 + e1;
            local_sum[1] += e2 + e3;
        }
        float l_ij0 = cute_fa_wgmma::row4_reduce_sum(local_sum[0]);
        float l_ij1 = cute_fa_wgmma::row4_reduce_sum(local_sum[1]);

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

        // Rescale O accumulator
        #pragma unroll
        for (int g = 0; g < MMA_N_PV; ++g) {
            tOrC(g*4+0) *= alpha0; tOrC(g*4+1) *= alpha0;
            tOrC(g*4+2) *= alpha1; tOrC(g*4+3) *= alpha1;
        }
        // Scale P
        #pragma unroll
        for (int g = 0; g < MMA_N_QK; ++g) {
            tSrC(g*4+0) *= beta0; tSrC(g*4+1) *= beta0;
            tSrC(g*4+2) *= beta1; tSrC(g*4+3) *= beta1;
        }

        // ---- Store P to sP (FP32 → FP16) ----
        #pragma unroll
        for (int g = 0; g < MMA_N_QK; ++g) {
            int col_base = g * 8 + 2 * (lane & 3);
            sP(row_hi_b, col_base)     = cute::half_t(tSrC(g*4+0));
            sP(row_hi_b, col_base + 1) = cute::half_t(tSrC(g*4+1));
            sP(row_lo_b, col_base)     = cute::half_t(tSrC(g*4+2));
            sP(row_lo_b, col_base + 1) = cute::half_t(tSrC(g*4+3));
        }
        cutlass::arch::NamedBarrier(WG_SIZE, 2).sync();

        // ---- WGMMA PV ----
        warpgroup_fence_operand(tOrC);
        warpgroup_arrive();
        gemm(mma_pv, tOsP, tOsVt(_, _, _, read_pipe), tOrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrC);

        // Signal producer: stage is free
        if (consumer_tid == 0) {
            EmptyBarrier::arrive(&empty_bar[read_pipe]);
        }

        ++read_pipe;
        if (read_pipe == NUM_STAGES) { read_pipe = 0; read_phase ^= 1; }
    }

    // ---- Epilogue ----
    float inv_l0 = (l_i[0] > 0.0f) ? 1.0f / l_i[0] : 0.0f;
    float inv_l1 = (l_i[1] > 0.0f) ? 1.0f / l_i[1] : 0.0f;

    const int r_hi_g = i0 + row_hi_b;
    const int r_lo_g = i0 + row_lo_b;
    #pragma unroll
    for (int g = 0; g < MMA_N_PV; ++g) {
        int col_base = g * 8 + 2 * (lane & 3);
        if (r_hi_g < seq_len) {
            mO(r_hi_g, col_base    ) = cute::half_t(tOrC(g*4+0) * inv_l0);
            mO(r_hi_g, col_base + 1) = cute::half_t(tOrC(g*4+1) * inv_l0);
        }
        if (r_lo_g < seq_len) {
            mO(r_lo_g, col_base    ) = cute::half_t(tOrC(g*4+2) * inv_l1);
            mO(r_lo_g, col_base + 1) = cute::half_t(tOrC(g*4+3) * inv_l1);
        }
    }
}


template <int Bc = 64, int D_MODEL = 128, int NUM_STAGES = 2>
void flash_attention_wgmma(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    assert(d_model == D_MODEL);
    constexpr int Br = 64;
    assert(seq_len % Br == 0 && seq_len % Bc == 0);

    using SmemAtomK  = GMMA::Layout_K_SW128_Atom<cute::half_t>;
    using SmemAtomMN = GMMA::Layout_MN_SW128_Atom<cute::half_t>;

    auto sQ_layout  = tile_to_shape(SmemAtomK{}, make_shape(Int<Br>{}, Int<D_MODEL>{}));
    auto sK_layout  = tile_to_shape(SmemAtomK{}, make_shape(Int<Bc>{}, Int<D_MODEL>{}, Int<NUM_STAGES>{}));
    // sVt: (D, Bc, STAGES) MN-major — TMA loads transposed V directly
    auto sVt_layout = tile_to_shape(SmemAtomMN{}, make_shape(Int<D_MODEL>{}, Int<Bc>{}, Int<NUM_STAGES>{}));
    auto sP_layout  = tile_to_shape(SmemAtomK{}, make_shape(Int<Br>{}, Int<Bc>{}));

    const int total_rows = batch_size * num_heads * seq_len;
    Tensor mQ_desc = make_tensor(Q, make_shape(total_rows, Int<D_MODEL>{}),
                                 make_stride(Int<D_MODEL>{}, _1{}));
    Tensor mK_desc = make_tensor(K, make_shape(total_rows, Int<D_MODEL>{}),
                                 make_stride(Int<D_MODEL>{}, _1{}));
    // Transposed V gmem: shape (D, total_rows), stride (1, D)
    // First mode (D) has stride 1 → matches MN-major smem
    Tensor mVt_desc = make_tensor(V, make_shape(Int<D_MODEL>{}, total_rows),
                                  make_stride(_1{}, Int<D_MODEL>{}));

    auto tma_q = make_tma_atom(SM90_TMA_LOAD{}, mQ_desc, sQ_layout,
                               make_shape(Int<Br>{}, Int<D_MODEL>{}));
    auto tma_k = make_tma_atom(SM90_TMA_LOAD{}, mK_desc, sK_layout(_, _, 0),
                               make_shape(Int<Bc>{}, Int<D_MODEL>{}));
    auto tma_vt = make_tma_atom(SM90_TMA_LOAD{}, mVt_desc, sVt_layout(_, _, 0),
                                make_shape(Int<D_MODEL>{}, Int<Bc>{}));

    // WGMMA for QK^T
    auto mma_qk = [&]() {
        if constexpr (Bc == 64) {
            return make_tiled_mma(
                SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else {
            return make_tiled_mma(
                SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        }
    }();

    // WGMMA for PV: A(sP) K-major, B(sVt) MN-major
    auto mma_pv = make_tiled_mma(
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>{});

    constexpr int sQ_size  = cosize_v<decltype(sQ_layout)>;
    constexpr int sK_size  = cosize_v<decltype(sK_layout)>;
    constexpr int sVt_size = cosize_v<decltype(sVt_layout)>;
    constexpr int sP_sz    = cosize_v<decltype(sP_layout)>;
    constexpr int smem_halves    = sQ_size + sK_size + sVt_size + sP_sz;
    constexpr int smem_bar_off   = (smem_halves * 2 + 127) & ~127;
    constexpr int smem_bar_bytes = (1 + 2 * NUM_STAGES) * sizeof(uint64_t);
    constexpr int smem_bytes     = smem_bar_off + smem_bar_bytes;

    auto kernel = flash_attention_wgmma_device<
        Bc, D_MODEL, NUM_STAGES,
        decltype(tma_q), decltype(tma_k), decltype(tma_vt),
        decltype(sQ_layout), decltype(sK_layout), decltype(sVt_layout),
        decltype(sP_layout),
        decltype(mma_qk), decltype(mma_pv)>;

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(256);
    kernel<<<grid, block, smem_bytes>>>(
        seq_len, total_rows, tma_q, tma_k, tma_vt,
        sQ_layout, sK_layout, sVt_layout,
        sP_layout,
        mma_qk, mma_pv,
        mask, out);
}
