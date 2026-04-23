#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention v2 (CUTLASS/CuTe, no tensor cores — UniversalFMA).
//
// Warp-per-row + online softmax, register-resident (l, m, O).
// Layout: [batch, heads, seq_len, d_model] row-major, cute::half_t storage, float compute.
//
// Grid : (ceil(seq_len / Br), batch * heads)
// Block: Br warps × 32 lanes

namespace cute_fa_register {

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

} // namespace cute_fa_register

template <int Br, int Bc, int D_MAX = 8>
__global__ static void flash_attention_register_device(
    int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    using namespace cute;
    static_assert(Bc % 32 == 0, "Bc must be a multiple of warpSize");
    constexpr int NUM_WARPS = Br;
    constexpr int ELTS_c    = Bc / 32;

    const int bh       = blockIdx.y;
    const int qkv_off  = bh * seq_len * d_model;
    const int mask_off = bh * seq_len * seq_len;
    const int i0       = blockIdx.x * Br;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int i       = i0 + warp_id;

    const int d_padded = ((d_model + 31) / 32) * 32 + 1;
    const float scale  = 1.0f / sqrtf((float)d_model);

    // Row-major gmem tensors (per batch*head).
    Tensor mQ = make_tensor(make_gmem_ptr(Q + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));
    Tensor mK = make_tensor(make_gmem_ptr(K + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));
    Tensor mV = make_tensor(make_gmem_ptr(V + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));
    Tensor mO = make_tensor(make_gmem_ptr(out + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));

    // Smem tiles: padded last dim to avoid bank conflicts.
    extern __shared__ __align__(16) char smem_raw[];
    cute::half_t* smem = reinterpret_cast<cute::half_t*>(smem_raw);
    cute::half_t* sQ_p = smem;
    cute::half_t* sK_p = sQ_p + Br * d_padded;
    cute::half_t* sV_p = sK_p + Bc * d_padded;

    Tensor sQ = make_tensor(make_smem_ptr(sQ_p),
                            make_layout(make_shape(Br, d_model), make_stride(d_padded, 1)));
    Tensor sK = make_tensor(make_smem_ptr(sK_p),
                            make_layout(make_shape(Bc, d_model), make_stride(d_padded, 1)));
    Tensor sV = make_tensor(make_smem_ptr(sV_p),
                            make_layout(make_shape(Bc, d_model), make_stride(d_padded, 1)));

    // TiledMMA (UniversalFMA) — declared for the QK^T inner tile. Used for
    // typing/documentation; the mainloop below is an explicit FMA loop so
    // fragment sizes stay small across varying d_model.
    [[maybe_unused]] auto tiled_mma = make_tiled_mma(
        cute::MMA_Atom<cute::UniversalFMA<float, cute::half_t, cute::half_t, float>>{},
        make_layout(make_shape(Int<Br>{}, Int<32>{})));

    // Per-warp output accumulator: O_acc[tc] holds one d-column chunk.
    float O_acc[D_MAX];
    #pragma unroll
    for (int d = 0; d < D_MAX; d++) O_acc[d] = 0.0f;

    // Load Q row (one warp per row, lanes split along d_model).
    for (int k = lane; k < d_model; k += 32) {
        sQ(warp_id, k) = (i < seq_len) ? mQ(i, k) : cute::half_t(0.0f);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        // Cooperative K / V tile load.
        for (int j = warp_id; j < Bc; j += NUM_WARPS) {
            for (int k = lane; k < d_model; k += 32) {
                sK(j, k) = (j0 + j < seq_len) ? mK(j0 + j, k) : cute::half_t(0.0f);
                sV(j, k) = (j0 + j < seq_len) ? mV(j0 + j, k) : cute::half_t(0.0f);
            }
        }
        __syncthreads();

        // S = Q_i · K_j^T   (per lane: ELTS_c score columns)
        float S[ELTS_c];
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) S[tc] = 0.0f;

        for (int k = 0; k < d_model; k++) {
            const float q = float(sQ(warp_id, k));
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                const int j = tc * 32 + lane;
                const float kv = (j < Bc) ? float(sK(j, k)) : 0.0f;
                S[tc] += q * kv;
            }
        }

        float local_max = -INFINITY;
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            S[tc] *= scale;
            const int j = tc * 32 + lane;
            if (mask && j0 + j < seq_len) {
                S[tc] += float(mask[mask_off + i * seq_len + (j0 + j)]);
            }
            local_max = fmaxf(local_max, S[tc]);
        }
        float m_ij = cute_fa_register::warp_reduce_max(local_max);
        if (!isfinite(m_ij)) { __syncthreads(); continue; }

        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) S[tc] = __expf(S[tc] - m_ij);

        float local_sum = 0.0f;
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) local_sum += S[tc];
        float l_ij = cute_fa_register::warp_reduce_sum(local_sum);

        const float m_i_new = fmaxf(m_i, m_ij);
        const float alpha   = __expf(m_i  - m_i_new);
        const float beta    = __expf(m_ij - m_i_new);
        const float l_i_new = alpha * l_i + beta * l_ij;

        // O_acc = (l_i * alpha * O_acc + beta * P V) / l_i_new   (per-d chunk)
        for (int d = lane; d < d_model; d += 32) {
            float pv = 0.0f;
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                for (int s = 0; s < 32; s++) {
                    const float P = __shfl_sync(0xffffffff, S[tc], s);
                    pv += P * float(sV(tc * 32 + s, d));
                }
            }
            if (i < seq_len) {
                O_acc[d / 32] = (l_i * alpha * O_acc[d / 32] + beta * pv) / l_i_new;
            }
        }

        m_i = m_i_new;
        l_i = l_i_new;
        __syncthreads();
    }

    for (int d = lane; d < d_model; d += 32) {
        if (i < seq_len) mO(i, d) = cute::half_t(O_acc[d / 32]);
    }
}


template <int Bc = 32, int Br = 8>
void flash_attention_register(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    const int d_padded = ((d_model + 31) / 32) * 32 + 1;
    const size_t smem  = (Br + 2 * Bc) * d_padded * sizeof(cute::half_t);

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(Br * 32);

    flash_attention_register_device<Br, Bc><<<grid, block, smem>>>(
        seq_len, d_model, Q, K, V, mask, out);
}
