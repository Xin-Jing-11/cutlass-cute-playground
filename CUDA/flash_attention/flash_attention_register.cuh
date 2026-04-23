#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention v2 (forward), mixed precision: __half storage, float compute.
//
// Layout (same as v1): Q, K, V, mask, out are [batch, heads, seq_len, d_model], row-major.
// One warp handles one query row; online softmax accumulates across J-tiles.
// D_MAX must be >= ceil(d_model / warpSize); default 8 supports d_model up to 256.

namespace fa_detail {

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

}  // namespace fa_detail


template <int NUM_WARPS_PER_BLOCK, int Br, int Bc, int D_MAX>
__global__ void flash_attention_register_kernel(
    const __half* Q,
    const __half* K,
    const __half* V,
    const __half* mask,
    __half* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_model
) {
    using fa_detail::warp_reduce_max;
    using fa_detail::warp_reduce_sum;

    static_assert(Bc % 32 == 0, "Bc must be a multiple of warpSize");
    static_assert(NUM_WARPS_PER_BLOCK == Br, "NUM_WARPS_PER_BLOCK must equal Br");

    const int d_model_padded = ((d_model + 31) / 32) * 32 + 1;
    const int i0 = blockIdx.x * Br;
    const int offset = blockIdx.y * seq_len * d_model;
    const int offset_mask = blockIdx.y * seq_len * seq_len;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int i = warp_id + i0;

    constexpr int warpSize_ = 32;
    constexpr int ELTS_c = Bc / warpSize_;

    float O_acc[D_MAX];
    #pragma unroll
    for (int d = 0; d < D_MAX; d++) O_acc[d] = 0.0f;

    extern __shared__ __align__(16) char smem_bytes[];
    __half* smem = reinterpret_cast<__half*>(smem_bytes);
    __half* Q_tiles = smem;
    __half* K_tiles = smem + Br * d_model_padded;
    __half* V_tiles = smem + (Bc + Br) * d_model_padded;
    const float scale = 1.0f / sqrtf((float)d_model);

    for (int k = lane_id; k < d_model; k += warpSize_) {
        Q_tiles[warp_id * d_model_padded + k] =
            (i < seq_len) ? __ldg(&Q[offset + i * d_model + k]) : __float2half(0.0f);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        for (int j = warp_id; j < Bc; j += NUM_WARPS_PER_BLOCK) {
            for (int k = lane_id; k < d_model; k += warpSize_) {
                K_tiles[j * d_model_padded + k] =
                    (j0 + j < seq_len) ? __ldg(&K[offset + (j0 + j) * d_model + k]) : __float2half(0.0f);
                V_tiles[j * d_model_padded + k] =
                    (j0 + j < seq_len) ? __ldg(&V[offset + (j0 + j) * d_model + k]) : __float2half(0.0f);
            }
        }
        __syncthreads();

        float S[ELTS_c];
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) S[tc] = 0.0f;

        for (int k = 0; k < d_model; k++) {
            const float q_ik = __half2float(Q_tiles[warp_id * d_model_padded + k]);
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                const int j = tc * warpSize_ + lane_id;
                const float k_jk = (j < Bc) ? __half2float(K_tiles[j * d_model_padded + k]) : 0.0f;
                S[tc] += q_ik * k_jk;
            }
        }

        float local_max_S = -INFINITY;
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            S[tc] *= scale;
            const int j = tc * warpSize_ + lane_id;
            if (mask && j0 + j < seq_len) {
                S[tc] += __half2float(__ldg(&mask[offset_mask + i * seq_len + (j0 + j)]));
            }
            local_max_S = fmaxf(local_max_S, S[tc]);
        }

        float m_ij = warp_reduce_max(local_max_S);

        if (!isfinite(m_ij)) continue;

        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            S[tc] = __expf(S[tc] - m_ij);
        }

        float local_sum_S = 0.0f;
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) local_sum_S += S[tc];
        float l_ij = warp_reduce_sum(local_sum_S);

        const float m_i_new = fmaxf(m_i, m_ij);
        const float alpha = __expf(m_i - m_i_new);
        const float beta = __expf(m_ij - m_i_new);
        const float l_i_new = alpha * l_i + beta * l_ij;

        for (int d = lane_id; d < d_model; d += warpSize_) {
            float O_id = 0.0f;
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                for (int s = 0; s < warpSize_; s++) {
                    const float P_ij = __shfl_sync(0xffffffff, S[tc], s);
                    O_id += P_ij * __half2float(V_tiles[(tc * warpSize_ + s) * d_model_padded + d]);
                }
            }
            if (i < seq_len) {
                O_acc[d / warpSize_] = (l_i * alpha * O_acc[d / warpSize_] + beta * O_id) / l_i_new;
            }
        }

        m_i = m_i_new;
        l_i = l_i_new;
        __syncthreads();
    }

    for (int d = lane_id; d < d_model; d += warpSize_) {
        if (i < seq_len) {
            out[offset + i * d_model + d] = __float2half(O_acc[d / warpSize_]);
        }
    }
}


template <int Bc = 32, int Br = 8, int D_MAX = 8>
void flash_attention_register(
    const __half* Q, const __half* K, const __half* V, const __half* mask,
    __half* out, int batch_size, int num_heads, int seq_len, int d_model
) {
    constexpr int NUM_WARPS_PER_BLOCK = Br;
    const int d_model_padded = ((d_model + 31) / 32) * 32 + 1;
    const size_t smem_bytes = (Br + 2 * Bc) * d_model_padded * sizeof(__half);

    dim3 grid((seq_len + Br - 1) / Br, batch_size * num_heads);
    dim3 block(NUM_WARPS_PER_BLOCK * 32);
    flash_attention_register_kernel<NUM_WARPS_PER_BLOCK, Br, Bc, D_MAX><<<grid, block, smem_bytes>>>(
        Q, K, V, mask, out, batch_size, num_heads, seq_len, d_model);
}
