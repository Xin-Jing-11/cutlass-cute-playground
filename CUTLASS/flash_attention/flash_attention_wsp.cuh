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

// Flash Attention "wsp" — warp-specialized + TMA loads + P in registers.
//
// Block layout:
//   warp 0          = producer (32 threads). Elect_one issues TMA for Q (once) and
//                     K+V (per j-tile). TMA auto-arrives on ClusterTransactionBarrier
//                     when the HW transaction bytes land — producer NEVER blocks on
//                     its own loads, just issues the next one.
//   warps 1..N      = consumers (N = Br/16 warps). Wait on FullBarrier, do
//                     transpose(V) + QK^T + softmax + PV in registers, arrive on
//                     EmptyBarrier to free the stage for the producer.
//
// Requirements: Br % 16 == 0, Bc % 16 == 0, D_MODEL % 16 == 0, D_MODEL % 8 == 0,
//               seq_len % Br == 0, seq_len % Bc == 0, NUM_STAGES >= 2.

namespace cute_fa_wsp {

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

} // namespace cute_fa_wsp


template <int Br, int Bc, int D_MODEL, int NUM_STAGES,
          class TmaQ, class TmaK, class TmaV,
          class SQLayout, class SKLayout, class SVrLayout>
__global__ static
__launch_bounds__(((Br / 16) + 1) * 32)
void flash_attention_wsp_device(
    int seq_len, int total_rows,
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    SQLayout sQ_layout, SKLayout sK_layout, SVrLayout sVr_layout,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    static_assert(Br % 16 == 0);
    static_assert(Bc % 16 == 0);
    static_assert(D_MODEL % 16 == 0);
    static_assert(NUM_STAGES >= 2);

    constexpr int CONSUMER_WARPS   = Br / 16;
    constexpr int CONSUMER_THREADS = CONSUMER_WARPS * 32;
    constexpr int PRODUCER_THREADS = 32;

    const int bh       = blockIdx.y;
    const int mask_off = bh * seq_len * seq_len;
    const int i0       = blockIdx.x * Br;

    const int tid          = threadIdx.x;
    const int warp_id      = tid >> 5;
    const int lane         = tid & 31;
    const bool is_producer = (warp_id == 0);
    const int consumer_tid = tid - PRODUCER_THREADS;

    const int consumer_warp_id = is_producer ? 0 : (consumer_tid >> 5);
    const int row_hi_w         = lane >> 2;
    const int row_lo_w         = row_hi_w + 8;
    const int row_hi_b         = consumer_warp_id * 16 + row_hi_w;
    const int row_lo_b         = consumer_warp_id * 16 + row_lo_w;

    const float scale = 1.0f / sqrtf((float)D_MODEL);
    constexpr int MMA_N_QK = Bc / 8;
    constexpr int MMA_N_PV = D_MODEL / 8;

    Tensor mO = make_tensor(make_gmem_ptr(out + bh * seq_len * D_MODEL),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));

    // -------- Shared memory --------
    extern __shared__ __align__(128) char smem_raw[];
    cute::half_t* smem_base = reinterpret_cast<cute::half_t*>(smem_raw);

    constexpr int sQ_size  = cosize_v<decltype(sQ_layout)>;
    constexpr int sK_size  = cosize_v<decltype(sK_layout)>;     // (Bc, D, STAGES)
    constexpr int sVr_size = cosize_v<decltype(sVr_layout)>;    // (Bc, D, STAGES)
    constexpr int sV_size  = D_MODEL * Bc;                      // transposed, single buf

    cute::half_t* sQ_p  = smem_base;
    cute::half_t* sK_p  = sQ_p  + sQ_size;
    cute::half_t* sVr_p = sK_p  + sK_size;
    cute::half_t* sV_p  = sVr_p + sVr_size;

    constexpr int smem_halves  = sQ_size + sK_size + sVr_size + sV_size;
    constexpr int smem_bar_off = (smem_halves * 2 + 15) & ~15;
    uint64_t* q_bar     = reinterpret_cast<uint64_t*>(smem_raw + smem_bar_off);
    uint64_t* kv_bar    = q_bar + 1;
    uint64_t* empty_bar = kv_bar + NUM_STAGES;

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_p),  sQ_layout);            // (Br, D)
    Tensor sK  = make_tensor(make_smem_ptr(sK_p),  sK_layout);            // (Bc, D, STAGES)
    Tensor sVr = make_tensor(make_smem_ptr(sVr_p), sVr_layout);           // (Bc, D, STAGES)

    auto sV_layout = make_layout(make_shape(Int<D_MODEL>{}, Int<Bc>{}),
                                 make_stride(Int<Bc>{}, _1{}));
    Tensor sV = make_tensor(make_smem_ptr(sV_p), sV_layout);

    auto sS_dummy_layout = make_layout(make_shape(Int<Br>{}, Int<Bc>{}),
                                       make_stride(Int<Bc>{}, _1{}));
    Tensor sS_dummy = make_tensor(make_smem_ptr(sQ_p), sS_dummy_layout);
    Tensor sO_dummy = make_tensor(make_smem_ptr(sQ_p),
                                  make_layout(make_shape(Int<Br>{}, Int<D_MODEL>{}),
                                              make_stride(Int<D_MODEL>{}, _1{})));

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    // -------- Init barriers (thread 0) --------
    if (tid == 0) {
        FullBarrier::init(q_bar, 1);
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            FullBarrier::init(&kv_bar[s],    1);
            EmptyBarrier::init(&empty_bar[s], CONSUMER_THREADS);
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
    // PRODUCER (warp 0)
    // ============================================================
    if (is_producer) {
        bool is_leader = (lane == 0);

        // Q TMA (once)
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

        // K, V TMA partitions.
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

        // Prologue: fill all NUM_STAGES stages.
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

        int write_pipe  = 0;
        int write_phase = 0;

        // Mainloop.
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

    // ============================================================
    // CONSUMER (warps 1..N)
    // ============================================================

    // Wait for Q to land.
    FullBarrier::wait(q_bar, 0);

    // Tiled MMA over consumer warps only.
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<CONSUMER_WARPS>, _1, _1>>{});
    auto thr_mma = tiled_mma.get_slice(consumer_tid);

    Tensor tSrA = thr_mma.partition_fragment_A(sQ);
    Tensor tSrB = thr_mma.partition_fragment_B(sK(_, _, 0));
    Tensor tSrC = thr_mma.partition_fragment_C(sS_dummy);

    Tensor tOrA = thr_mma.partition_fragment_A(sS_dummy);
    Tensor tOrB = thr_mma.partition_fragment_B(sV);
    Tensor tOrC = thr_mma.partition_fragment_C(sO_dummy);
    clear(tOrC);

    auto s2r_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma);

    auto s2r_thr_Q = s2r_copy_Q.get_slice(consumer_tid);
    auto s2r_thr_K = s2r_copy_K.get_slice(consumer_tid);
    auto s2r_thr_V = s2r_copy_V.get_slice(consumer_tid);

    Tensor tXsQ = s2r_thr_Q.partition_S(sQ);
    Tensor tXrQ = s2r_thr_Q.retile_D(tSrA);
    Tensor tXrK = s2r_thr_K.retile_D(tSrB);
    Tensor tXsV = s2r_thr_V.partition_S(sV);
    Tensor tXrV = s2r_thr_V.retile_D(tOrB);

    float m_i[2] = { -INFINITY, -INFINITY };
    float l_i[2] = {  0.0f,      0.0f     };

    int read_pipe  = 0;
    int read_phase = 0;

    for (int tile = 0; tile < T_c; ++tile) {
        int j0 = tile * Bc;

        // Wait for TMA.
        FullBarrier::wait(&kv_bar[read_pipe], read_phase);

        // Transpose sVr[read_pipe] → sV.
        Tensor sVr_cur = sVr(_, _, read_pipe);
        #pragma unroll
        for (int idx = consumer_tid; idx < Bc * D_MODEL; idx += CONSUMER_THREADS) {
            int r = idx / D_MODEL;
            int c = idx % D_MODEL;
            sV(c, r) = sVr_cur(r, c);
        }
        cutlass::arch::NamedBarrier(CONSUMER_THREADS, 1).sync();

        Tensor sK_cur = sK(_, _, read_pipe);
        Tensor tXsK_cur = s2r_thr_K.partition_S(sK_cur);

        // ---- QK^T MMA ----
        clear(tSrC);
        #pragma unroll
        for (int k = 0; k < size<2>(tSrA); ++k) {
            copy(s2r_copy_Q, tXsQ(_, _, k),     tXrQ(_, _, k));
            copy(s2r_copy_K, tXsK_cur(_, _, k), tXrK(_, _, k));
            gemm(tiled_mma, tSrA(_, _, k), tSrB(_, _, k), tSrC);
        }

        // ---- Softmax ----
        float local_max[2] = { -INFINITY, -INFINITY };
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
                int r_hi_g = i0 + row_hi_b;
                int r_lo_g = i0 + row_lo_b;
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
        float m_ij0 = cute_fa_wsp::row4_reduce_max(local_max[0]);
        float m_ij1 = cute_fa_wsp::row4_reduce_max(local_max[1]);

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
        float l_ij0 = cute_fa_wsp::row4_reduce_sum(local_sum[0]);
        float l_ij1 = cute_fa_wsp::row4_reduce_sum(local_sum[1]);

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

        // P register-to-register.
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
            gemm(tiled_mma, tOrA(_, _, k), tOrB(_, _, k), tOrC);
        }

        // Signal producer that this stage is free.
        EmptyBarrier::arrive(&empty_bar[read_pipe]);
        // Consumer-only sync so sV reads are done before next iter's transpose.
        cutlass::arch::NamedBarrier(CONSUMER_THREADS, 1).sync();

        ++read_pipe;
        if (read_pipe == NUM_STAGES) { read_pipe = 0; read_phase ^= 1; }
    }

    // Epilogue.
    float inv_l0 = (l_i[0] > 0.0f) ? 1.0f / l_i[0] : 0.0f;
    float inv_l1 = (l_i[1] > 0.0f) ? 1.0f / l_i[1] : 0.0f;

    const int r_hi_g = i0 + row_hi_b;
    const int r_lo_g = i0 + row_lo_b;
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


template <int Bc = 32, int Br = 64, int D_MODEL = 128, int NUM_STAGES = 2>
void flash_attention_wsp(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    assert(d_model == D_MODEL);
    assert(seq_len % Br == 0 && seq_len % Bc == 0);

    constexpr int CONSUMER_WARPS = Br / 16;
    constexpr int TOTAL_THREADS  = (CONSUMER_WARPS + 1) * 32;

    // GMMA K-major 128-byte swizzle — TMA-native, ldmatrix-compatible.
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
    constexpr int smem_halves    = sQ_size + sK_size + sVr_size + sV_size;
    constexpr int smem_bar_off   = (smem_halves * 2 + 15) & ~15;
    constexpr int smem_bar_bytes = (1 + 2 * NUM_STAGES) * sizeof(uint64_t);
    constexpr int smem_bytes     = smem_bar_off + smem_bar_bytes;

    auto kernel = flash_attention_wsp_device<
        Br, Bc, D_MODEL, NUM_STAGES,
        decltype(tma_q), decltype(tma_k), decltype(tma_v),
        decltype(sQ_layout), decltype(sK_layout), decltype(sVr_layout)>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(TOTAL_THREADS);
    kernel<<<grid, block, smem_bytes>>>(
        seq_len, total_rows, tma_q, tma_k, tma_v,
        sQ_layout, sK_layout, sVr_layout,
        mask, out);
}
