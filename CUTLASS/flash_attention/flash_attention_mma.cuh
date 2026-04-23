#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention "mma" — tensor-core SM80_16x8x16_F32F16F16F32_TN for both QK^T and PV.
//
// Layout: [batch, heads, seq_len, d_model] row-major, cute::half_t storage, float compute.
// Br must be a multiple of 16; NUM_WARPS = Br / 16 warps per CTA each handle 16 query rows.
// All warps cooperatively load Q, K, V into shared memory and share those tiles across
// the per-warp softmax accumulators. Online softmax uses the unnormalized update
// (single division deferred to epilogue). P materialized in smem between the two MMAs.
//
// Requirements:
//   Br % 16 == 0             (MMA_M of the atom × NUM_WARPS)
//   Bc % 16 == 0             (MMA_K=16 for PV)
//   D_MODEL % 16 == 0        (MMA_K=16 for QK^T)
//   seq_len % Br == 0        (no tail handling in this first version)

namespace cute_fa_mma {

// 4-thread row-group reduction: lanes {4k, 4k+1, 4k+2, 4k+3} share a row.
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

} // namespace cute_fa_mma


template <int Br, int Bc, int D_MODEL>
__global__ static void flash_attention_mma_device(
    int seq_len,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    static_assert(Br % 16 == 0, "Br must be multiple of 16 (MMA_M of 16x8x16 atom)");
    static_assert(Bc % 16 == 0, "Bc must be multiple of 16");
    static_assert(D_MODEL % 16 == 0, "D_MODEL must be multiple of 16");

    constexpr int NUM_WARPS   = Br / 16;
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    const int bh       = blockIdx.y;
    const int qkv_off  = bh * seq_len * D_MODEL;
    const int mask_off = bh * seq_len * seq_len;
    const int i0       = blockIdx.x * Br;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;                   // 0..NUM_WARPS-1
    const int lane    = tid & 31;
    // Each thread owns two rows *within its warp's 16-row block*.
    const int row_hi_w = lane >> 2;                 // 0..7 (within warp's 16 rows)
    const int row_lo_w = row_hi_w + 8;              // 8..15 (within warp's 16 rows)
    const int row_hi_b = warp_id * 16 + row_hi_w;   // 0..Br-1 (within the CTA's Br block)
    const int row_lo_b = warp_id * 16 + row_lo_w;

    const float scale = 1.0f / sqrtf((float)D_MODEL);

    constexpr int MMA_N_QK = Bc / 8;       // n-tiles for QK^T
    constexpr int MMA_N_PV = D_MODEL / 8;  // n-tiles for PV

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

    // -------- Shared memory: sQ, sK, sV (transposed), sP ---------
    extern __shared__ __align__(16) char smem_raw[];
    cute::half_t* sQ_p = reinterpret_cast<cute::half_t*>(smem_raw);
    cute::half_t* sK_p = sQ_p + Br      * D_MODEL;
    cute::half_t* sV_p = sK_p + Bc      * D_MODEL;
    cute::half_t* sP_p = sV_p + D_MODEL * Bc;

    Tensor sQ = make_tensor(make_smem_ptr(sQ_p),
                            make_layout(make_shape(Int<Br>{},      Int<D_MODEL>{}),
                                        make_stride(Int<D_MODEL>{}, _1{})));
    Tensor sK = make_tensor(make_smem_ptr(sK_p),
                            make_layout(make_shape(Int<Bc>{},       Int<D_MODEL>{}),
                                        make_stride(Int<D_MODEL>{}, _1{})));
    Tensor sV = make_tensor(make_smem_ptr(sV_p),
                            make_layout(make_shape(Int<D_MODEL>{}, Int<Bc>{}),
                                        make_stride(Int<Bc>{},      _1{})));
    Tensor sP = make_tensor(make_smem_ptr(sP_p),
                            make_layout(make_shape(Int<Br>{},       Int<Bc>{}),
                                        make_stride(Int<Bc>{},      _1{})));

    // A "shape-only" dummy for partition_fragment_C on the PV output (Br, D_MODEL).
    // Backing pointer is irrelevant — only the layout is used for fragment derivation.
    Tensor sO_dummy = make_tensor(make_smem_ptr(sP_p),
                                  make_layout(make_shape(Int<Br>{},       Int<D_MODEL>{}),
                                              make_stride(Int<D_MODEL>{}, _1{})));

    // -------- Tiled MMA: NUM_WARPS atoms tiled in M --------
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<NUM_WARPS>, _1, _1>>{});
    auto thr_mma = tiled_mma.get_slice(tid);

    // QK^T fragments
    Tensor tSrA = thr_mma.partition_fragment_A(sQ);    // (MMA, MMA_M=1, MMA_K=D_MODEL/16)
    Tensor tSrB = thr_mma.partition_fragment_B(sK);    // (MMA, MMA_N=Bc/8, MMA_K=D_MODEL/16)
    Tensor tSrC = thr_mma.partition_fragment_C(sP);    // (MMA=4, MMA_M=1, MMA_N=Bc/8) FP32

    // PV fragments
    Tensor tOrA = thr_mma.partition_fragment_A(sP);    // (MMA, MMA_M=1, MMA_K=Bc/16)
    Tensor tOrB = thr_mma.partition_fragment_B(sV);    // (MMA, MMA_N=D_MODEL/8, MMA_K=Bc/16)
    Tensor tOrC = thr_mma.partition_fragment_C(sO_dummy); // (MMA=4, MMA_M=1, MMA_N=D_MODEL/8) FP32
    clear(tOrC);

    // -------- ldmatrix copy atoms --------
    auto s2r_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_P = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
    auto s2r_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>{}, tiled_mma);

    auto s2r_thr_Q = s2r_copy_Q.get_slice(tid);
    auto s2r_thr_K = s2r_copy_K.get_slice(tid);
    auto s2r_thr_P = s2r_copy_P.get_slice(tid);
    auto s2r_thr_V = s2r_copy_V.get_slice(tid);

    Tensor tXsQ = s2r_thr_Q.partition_S(sQ);
    Tensor tXrQ = s2r_thr_Q.retile_D(tSrA);
    Tensor tXsK = s2r_thr_K.partition_S(sK);
    Tensor tXrK = s2r_thr_K.retile_D(tSrB);
    Tensor tXsP = s2r_thr_P.partition_S(sP);
    Tensor tXrP = s2r_thr_P.retile_D(tOrA);
    Tensor tXsV = s2r_thr_V.partition_S(sV);
    Tensor tXrV = s2r_thr_V.retile_D(tOrB);

    // -------- Cooperative load: Q tile (persistent across j-loop) --------
    constexpr int Q_ELTS = Br * D_MODEL;
    #pragma unroll
    for (int idx = tid; idx < Q_ELTS; idx += NUM_THREADS) {
        int r = idx / D_MODEL;
        int c = idx % D_MODEL;
        int row = i0 + r;
        sQ_p[r * D_MODEL + c] = (row < seq_len) ? mQ(row, c) : cute::half_t(0.0f);
    }
    __syncthreads();

    // Per-row online-softmax stats (each lane owns row_hi and row_lo).
    float m_i[2] = { -INFINITY, -INFINITY };
    float l_i[2] = {  0.0f,      0.0f     };

    // -------- Mainloop over j-tiles --------
    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        // Load K tile (natural layout) and V tile (transposed) into smem.
        constexpr int K_ELTS = Bc * D_MODEL;
        #pragma unroll
        for (int idx = tid; idx < K_ELTS; idx += NUM_THREADS) {
            int r = idx / D_MODEL;
            int c = idx % D_MODEL;
            int row = j0 + r;
            sK_p[r * D_MODEL + c] = (row < seq_len) ? mK(row, c) : cute::half_t(0.0f);
        }
        #pragma unroll
        for (int idx = tid; idx < K_ELTS; idx += NUM_THREADS) {
            int r = idx / D_MODEL;
            int c = idx % D_MODEL;
            int row = j0 + r;
            sV_p[c * Bc + r] = (row < seq_len) ? mV(row, c) : cute::half_t(0.0f);
        }
        __syncthreads();

        // ---- QK^T MMA ----
        clear(tSrC);
        #pragma unroll
        for (int k = 0; k < size<2>(tSrA); ++k) {
            copy(s2r_copy_Q, tXsQ(_, _, k), tXrQ(_, _, k));
            copy(s2r_copy_K, tXsK(_, _, k), tXrK(_, _, k));
            gemm(tiled_mma, tSrA(_, _, k), tSrB(_, _, k), tSrC);
        }

        // ---- Fragment-aware softmax on tSrC ----
        // tSrC per-thread layout for each (0, n):
        //   tSrC(0, 0, n) = S[row_hi, 2*(lane%4)    ]
        //   tSrC(1, 0, n) = S[row_hi, 2*(lane%4) + 1]
        //   tSrC(2, 0, n) = S[row_lo, 2*(lane%4)    ]
        //   tSrC(3, 0, n) = S[row_lo, 2*(lane%4) + 1]
        //   — shifted by n*8 along the column axis.

        // Scale and masking.
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

            tSrC(0, 0, n) = s0;
            tSrC(1, 0, n) = s1;
            tSrC(2, 0, n) = s2;
            tSrC(3, 0, n) = s3;

            local_max[0] = fmaxf(local_max[0], fmaxf(s0, s1));
            local_max[1] = fmaxf(local_max[1], fmaxf(s2, s3));
        }
        float m_ij0 = cute_fa_mma::row4_reduce_max(local_max[0]);
        float m_ij1 = cute_fa_mma::row4_reduce_max(local_max[1]);

        // exp and row-sum.
        float local_sum[2] = { 0.0f, 0.0f };
        #pragma unroll
        for (int n = 0; n < MMA_N_QK; ++n) {
            float e0 = isfinite(m_ij0) ? __expf(tSrC(0, 0, n) - m_ij0) : 0.0f;
            float e1 = isfinite(m_ij0) ? __expf(tSrC(1, 0, n) - m_ij0) : 0.0f;
            float e2 = isfinite(m_ij1) ? __expf(tSrC(2, 0, n) - m_ij1) : 0.0f;
            float e3 = isfinite(m_ij1) ? __expf(tSrC(3, 0, n) - m_ij1) : 0.0f;
            tSrC(0, 0, n) = e0;
            tSrC(1, 0, n) = e1;
            tSrC(2, 0, n) = e2;
            tSrC(3, 0, n) = e3;
            local_sum[0] += e0 + e1;
            local_sum[1] += e2 + e3;
        }
        float l_ij0 = cute_fa_mma::row4_reduce_sum(local_sum[0]);
        float l_ij1 = cute_fa_mma::row4_reduce_sum(local_sum[1]);

        // Online-softmax update.
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

        // Rescale tOrC (O accumulator) by alpha on a per-row basis.
        #pragma unroll
        for (int n = 0; n < MMA_N_PV; ++n) {
            tOrC(0, 0, n) *= alpha0;
            tOrC(1, 0, n) *= alpha0;
            tOrC(2, 0, n) *= alpha1;
            tOrC(3, 0, n) *= alpha1;
        }

        // Apply beta to P before storing (absorbs the beta factor into P directly).
        #pragma unroll
        for (int n = 0; n < MMA_N_QK; ++n) {
            tSrC(0, 0, n) *= beta0;
            tSrC(1, 0, n) *= beta0;
            tSrC(2, 0, n) *= beta1;
            tSrC(3, 0, n) *= beta1;
        }

        // Store P (FP16) to sP via the C-partition of sP.
        Tensor tPsP = thr_mma.partition_C(sP);
        #pragma unroll
        for (int n = 0; n < MMA_N_QK; ++n) {
            tPsP(0, 0, n) = cute::half_t(tSrC(0, 0, n));
            tPsP(1, 0, n) = cute::half_t(tSrC(1, 0, n));
            tPsP(2, 0, n) = cute::half_t(tSrC(2, 0, n));
            tPsP(3, 0, n) = cute::half_t(tSrC(3, 0, n));
        }
        __syncthreads();

        // ---- PV MMA: tOrC += P · V_T ----
        #pragma unroll
        for (int k = 0; k < size<2>(tOrA); ++k) {
            copy(s2r_copy_P, tXsP(_, _, k), tXrP(_, _, k));
            copy(s2r_copy_V, tXsV(_, _, k), tXrV(_, _, k));
            gemm(tiled_mma, tOrA(_, _, k), tOrB(_, _, k), tOrC);
        }
        __syncthreads();
    }

    // -------- Epilogue: divide by l_i, convert to FP16, write to gmem --------
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


template <int Bc = 32, int Br = 16, int D_MODEL = 64>
void flash_attention_mma(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    assert(d_model == D_MODEL && "flash_attention_mma requires d_model == D_MODEL template arg");
    constexpr int NUM_WARPS = Br / 16;
    const size_t smem_bytes = sizeof(cute::half_t) *
        (Br * D_MODEL + Bc * D_MODEL + D_MODEL * Bc + Br * Bc);

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(NUM_WARPS * 32);

    auto kernel = flash_attention_mma_device<Br, Bc, D_MODEL>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
    kernel<<<grid, block, smem_bytes>>>(seq_len, Q, K, V, mask, out);
}
