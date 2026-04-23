#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention "pregs" — multistage + P kept in registers (no sP round-trip).
//
// Same as flash_attention_multistage.cuh except: the QK^T accumulator fragment
// (tSrC, FP32) is converted directly to the PV-MMA A fragment (tOrA, FP16) in
// registers, eliminating the store-to-sP, __syncthreads, and ldmatrix-from-sP
// that multistage does. The register mapping exploits the fact that the MMA
// output C-fragment layout and the A-input fragment layout share the same
// (row, col) thread ownership: for each PV k-tile covering 16 S columns, the
// 8 halves of A come from 2 consecutive S n-tile outputs.
//
// Requirements:
//   Br % 16 == 0, Bc % 16 == 0, D_MODEL % 16 == 0, D_MODEL % 8 == 0
//   seq_len % Bc == 0  (non-predicated cp.async for simplicity)
//   NUM_STAGES >= 2

namespace cute_fa_pregs {

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

} // namespace cute_fa_pregs


template <int Br, int Bc, int D_MODEL, int NUM_STAGES>
__global__ static void flash_attention_pregs_device(
    int seq_len,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    static_assert(Br % 16 == 0,    "Br % 16 == 0");
    static_assert(Bc % 16 == 0,    "Bc % 16 == 0");
    static_assert(D_MODEL % 16 == 0, "D_MODEL % 16 == 0");
    static_assert(D_MODEL % 8  == 0, "D_MODEL % 8 == 0 (128-bit cp.async)");
    static_assert(NUM_STAGES >= 2, "NUM_STAGES >= 2");

    constexpr int NUM_WARPS   = Br / 16;
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    const int bh       = blockIdx.y;
    const int qkv_off  = bh * seq_len * D_MODEL;
    const int mask_off = bh * seq_len * seq_len;
    const int i0       = blockIdx.x * Br;

    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;
    const int row_hi_w = lane >> 2;
    const int row_lo_w = row_hi_w + 8;
    const int row_hi_b = warp_id * 16 + row_hi_w;
    const int row_lo_b = warp_id * 16 + row_lo_w;

    const float scale = 1.0f / sqrtf((float)D_MODEL);

    constexpr int MMA_N_QK = Bc / 8;
    constexpr int MMA_N_PV = D_MODEL / 8;

    Tensor mQ = make_tensor(make_gmem_ptr(Q + qkv_off),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(K + qkv_off),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(V + qkv_off),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));
    Tensor mO = make_tensor(make_gmem_ptr(out + qkv_off),
                            make_shape(seq_len, Int<D_MODEL>{}),
                            make_stride(Int<D_MODEL>{}, _1{}));

    // -------- Shared memory (no sP — P kept in registers) --------
    extern __shared__ __align__(16) char smem_raw[];
    cute::half_t* sQ_p  = reinterpret_cast<cute::half_t*>(smem_raw);
    cute::half_t* sK_base  = sQ_p + Br * D_MODEL;
    cute::half_t* sVr_base = sK_base + NUM_STAGES * Bc * D_MODEL;
    cute::half_t* sV_p     = sVr_base + NUM_STAGES * Bc * D_MODEL;

    auto sK_ptr  = [&](int s) { return sK_base  + s * Bc * D_MODEL; };
    auto sVr_ptr = [&](int s) { return sVr_base + s * Bc * D_MODEL; };

    // Swizzle<3,3,3> XORs bits [5:3] into bits [5:3]... scratch — use the standard
    // "K-contiguous with 8-half chunks" swizzle: XOR 3 row bits into 3 chunk bits.
    // For D_MODEL halves per row (128), row bit is at position log2(D_MODEL)=7, chunk
    // bits are at [5:3] (8-half chunks). Swizzle<3,3,3> on an atom of (8, D_MODEL)
    // base layout, tiled up to full shape, achieves this (CUTLASS-standard K-major
    // smem layout for 16x8x16 ldmatrix).
    using SwizzleKCon = Swizzle<3, 3, 3>;
    auto sQ_layout = tile_to_shape(
        composition(SwizzleKCon{},
            make_layout(make_shape(_8{}, Int<D_MODEL>{}),
                        make_stride(Int<D_MODEL>{}, _1{}))),
        make_shape(Int<Br>{}, Int<D_MODEL>{}));
    auto sK_layout = tile_to_shape(
        composition(SwizzleKCon{},
            make_layout(make_shape(_8{}, Int<D_MODEL>{}),
                        make_stride(Int<D_MODEL>{}, _1{}))),
        make_shape(Int<Bc>{}, Int<D_MODEL>{}));
    auto sVr_layout = sK_layout;  // same shape/layout as sK for cp.async target
    // sV is transposed ((D_MODEL, Bc)) with Bc-contiguous; swizzle for Bc=32.
    auto sV_layout = tile_to_shape(
        composition(SwizzleKCon{},
            make_layout(make_shape(_8{}, Int<Bc>{}),
                        make_stride(Int<Bc>{}, _1{}))),
        make_shape(Int<D_MODEL>{}, Int<Bc>{}));

    Tensor sQ = make_tensor(make_smem_ptr(sQ_p), sQ_layout);
    Tensor sV = make_tensor(make_smem_ptr(sV_p), sV_layout);
    // Shape-only dummies for partition_fragment_C (S: Br×Bc) and A (P: Br×Bc).
    // Backing ptr is irrelevant — CuTe only reads the shape to derive fragment layout.
    auto sS_dummy_layout = make_layout(make_shape(Int<Br>{}, Int<Bc>{}),
                                       make_stride(Int<Bc>{}, _1{}));
    Tensor sS_dummy = make_tensor(make_smem_ptr(sQ_p), sS_dummy_layout);
    Tensor sO_dummy = make_tensor(make_smem_ptr(sQ_p),
                                  make_layout(make_shape(Int<Br>{},       Int<D_MODEL>{}),
                                              make_stride(Int<D_MODEL>{}, _1{})));

    // -------- Tiled MMA --------
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<NUM_WARPS>, _1, _1>>{});
    auto thr_mma = tiled_mma.get_slice(tid);

    // QK^T fragments. sK "view" tensor (base updated per iter).
    Tensor sK0 = make_tensor(make_smem_ptr(sK_ptr(0)), sK_layout);
    Tensor tSrA = thr_mma.partition_fragment_A(sQ);
    Tensor tSrB = thr_mma.partition_fragment_B(sK0);
    Tensor tSrC = thr_mma.partition_fragment_C(sS_dummy);

    // PV A fragment (P in registers). Layout derived from a (Br, Bc) dummy.
    Tensor tOrA = thr_mma.partition_fragment_A(sS_dummy);
    Tensor tOrB = thr_mma.partition_fragment_B(sV);
    Tensor tOrC = thr_mma.partition_fragment_C(sO_dummy);
    clear(tOrC);

    // ldmatrix atoms (Q, K, V only — P stays in regs)
    auto s2r_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma);

    auto s2r_thr_Q = s2r_copy_Q.get_slice(tid);
    auto s2r_thr_K = s2r_copy_K.get_slice(tid);
    auto s2r_thr_V = s2r_copy_V.get_slice(tid);

    Tensor tXsQ = s2r_thr_Q.partition_S(sQ);
    Tensor tXrQ = s2r_thr_Q.retile_D(tSrA);
    Tensor tXrK = s2r_thr_K.retile_D(tSrB);
    Tensor tXsV = s2r_thr_V.partition_S(sV);
    Tensor tXrV = s2r_thr_V.retile_D(tOrB);

    // -------- cp.async tiled copy for (Bc, D_MODEL) tiles --------
    using G2SCopy = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>;
    constexpr int VEC  = 8;              // 8 halves = 128 bits
    constexpr int BK_V = D_MODEL / VEC;
    static_assert(NUM_THREADS % BK_V == 0, "NUM_THREADS must be divisible by D_MODEL/8");
    constexpr int ThrM_KV = NUM_THREADS / BK_V;
    static_assert(Bc % ThrM_KV == 0, "Bc must be divisible by NUM_THREADS/BK_V");

    auto g2s_copy = make_tiled_copy(G2SCopy{},
        make_layout(make_shape(Int<ThrM_KV>{}, Int<BK_V>{}), LayoutRight{}),
        make_layout(make_shape(_1{},            Int<VEC>{})));
    auto g2s_thr = g2s_copy.get_slice(tid);

    // Pre-slice the KV gmem view into tiles of Bc rows.
    Tensor mK_tiled = local_tile(mK, make_shape(Int<Bc>{}, Int<D_MODEL>{}), make_coord(_, 0));
    Tensor mV_tiled = local_tile(mV, make_shape(Int<Bc>{}, Int<D_MODEL>{}), make_coord(_, 0));
    //  shapes: (Bc, D_MODEL, T_c)

    // Lambda: issue cp.async for tile t of K into sK_ptr(stage), and V into sVr_ptr(stage).
    auto issue_load = [&](int t, int stage) {
        Tensor gK_t  = mK_tiled(_, _, t);
        Tensor gV_t  = mV_tiled(_, _, t);
        Tensor sK_st  = make_tensor(make_smem_ptr(sK_ptr(stage)),  sK_layout);
        Tensor sVr_st = make_tensor(make_smem_ptr(sVr_ptr(stage)), sVr_layout);
        Tensor tKgK = g2s_thr.partition_S(gK_t);
        Tensor tKsK = g2s_thr.partition_D(sK_st);
        Tensor tVgV = g2s_thr.partition_S(gV_t);
        Tensor tVsV = g2s_thr.partition_D(sVr_st);
        copy(g2s_copy, tKgK, tKsK);
        copy(g2s_copy, tVgV, tVsV);
    };

    // -------- Load Q via cp.async (persistent across j-loop) --------
    // Requires seq_len % Br == 0 so the Q block is fully in-range.
    {
        Tensor mQ_tiled = local_tile(mQ, make_shape(Int<Br>{}, Int<D_MODEL>{}), make_coord(_, 0));
        Tensor gQ_t = mQ_tiled(_, _, blockIdx.x);
        Tensor tQgQ = g2s_thr.partition_S(gQ_t);
        Tensor tQsQ = g2s_thr.partition_D(sQ);
        copy(g2s_copy, tQgQ, tQsQ);
    }

    // Fence Q load into its own group, then wait for it before starting the K/V pipeline.
    // This keeps the K/V pipeline's wait count matching NUM_STAGES exactly.
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // -------- Prologue: issue cp.async for stages 0..NUM_STAGES-2 --------
    const int T_c = (seq_len + Bc - 1) / Bc;
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        if (s < T_c) {
            issue_load(s, s);
        }
        cp_async_fence();
    }

    // Per-row softmax stats.
    float m_i[2] = { -INFINITY, -INFINITY };
    float l_i[2] = {  0.0f,      0.0f     };

    // -------- Mainloop --------
    for (int tile = 0; tile < T_c; ++tile) {
        int stage = tile % NUM_STAGES;
        int j0    = tile * Bc;

        // Wait for current stage's cp.async to land.
        cp_async_wait<NUM_STAGES - 2>();
        __syncthreads();

        // Synchronously transpose sVr[stage] → sV (MMA-ready), via CuTe indexing
        // so the swizzle is applied correctly.
        Tensor sVr_cur = make_tensor(make_smem_ptr(sVr_ptr(stage)), sVr_layout);
        #pragma unroll
        for (int idx = tid; idx < Bc * D_MODEL; idx += NUM_THREADS) {
            int r = idx / D_MODEL;
            int c = idx % D_MODEL;
            sV(c, r) = sVr_cur(r, c);
        }
        __syncthreads();

        // Issue next tile's load into the stage we just transposed out of.
        int next_tile = tile + NUM_STAGES - 1;
        if (next_tile < T_c) {
            issue_load(next_tile, next_tile % NUM_STAGES);
        }
        cp_async_fence();

        // Rebuild smem K tensor at current stage for ldmatrix source.
        Tensor sK_cur    = make_tensor(make_smem_ptr(sK_ptr(stage)), sK_layout);
        Tensor tXsK_cur  = s2r_thr_K.partition_S(sK_cur);

        // ---- QK^T MMA ----
        clear(tSrC);
        #pragma unroll
        for (int k = 0; k < size<2>(tSrA); ++k) {
            copy(s2r_copy_Q, tXsQ(_, _, k),     tXrQ(_, _, k));
            copy(s2r_copy_K, tXsK_cur(_, _, k), tXrK(_, _, k));
            gemm(tiled_mma, tSrA(_, _, k), tSrB(_, _, k), tSrC);
        }

        // ---- Softmax (same as mma) ----
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

            tSrC(0, 0, n) = s0; tSrC(1, 0, n) = s1;
            tSrC(2, 0, n) = s2; tSrC(3, 0, n) = s3;

            local_max[0] = fmaxf(local_max[0], fmaxf(s0, s1));
            local_max[1] = fmaxf(local_max[1], fmaxf(s2, s3));
        }
        float m_ij0 = cute_fa_pregs::row4_reduce_max(local_max[0]);
        float m_ij1 = cute_fa_pregs::row4_reduce_max(local_max[1]);

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
        float l_ij0 = cute_fa_pregs::row4_reduce_sum(local_sum[0]);
        float l_ij1 = cute_fa_pregs::row4_reduce_sum(local_sum[1]);

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
            tOrC(0, 0, n) *= alpha0;
            tOrC(1, 0, n) *= alpha0;
            tOrC(2, 0, n) *= alpha1;
            tOrC(3, 0, n) *= alpha1;
        }
        #pragma unroll
        for (int n = 0; n < MMA_N_QK; ++n) {
            tSrC(0, 0, n) *= beta0;
            tSrC(1, 0, n) *= beta0;
            tSrC(2, 0, n) *= beta1;
            tSrC(3, 0, n) *= beta1;
        }

        // ---- P: register-to-register (FP32 S-accum → FP16 A-fragment) ----
        // Layout mapping (for 16x8x16 TN):
        //   tSrC(0..3, 0, n) holds C-output elements at (row_hi, row_lo, even col, odd col)
        //     for the n-th 8-col MMA_N tile (covering S cols [n*8, n*8+8)).
        //   tOrA(0..7, 0, k_pv) holds A-input elements at the same (row, col) pattern,
        //     but each A-atom covers 16 cols: a[0..3] = cols 0..7 of the atom, a[4..7] = cols 8..15.
        // Thus for PV k-tile k_pv, take two consecutive S n-tiles (n = 2*k_pv and 2*k_pv+1).
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
        __syncthreads();
    }

    // -------- Epilogue --------
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
void flash_attention_pregs(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    assert(d_model == D_MODEL);
    constexpr int NUM_WARPS = Br / 16;
    const size_t smem_bytes = sizeof(cute::half_t) *
        (Br * D_MODEL                      // sQ
         + NUM_STAGES * Bc * D_MODEL       // sK stages
         + NUM_STAGES * Bc * D_MODEL       // sVr stages (natural)
         + D_MODEL * Bc);                  // sV transposed (single); no sP — P in regs

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(NUM_WARPS * 32);

    auto kernel = flash_attention_pregs_device<Br, Bc, D_MODEL, NUM_STAGES>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
    kernel<<<grid, block, smem_bytes>>>(seq_len, Q, K, V, mask, out);
}
