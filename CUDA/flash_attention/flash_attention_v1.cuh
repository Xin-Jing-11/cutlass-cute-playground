#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

/**
 * The paper implementation of FlashAttention v1, mixed precision.
 *
 * Storage:  Q, K, V, mask, out are __half (FP16).
 * Compute:  softmax stats (m, l, S) and accumulators are FP32.
 *
 * Conversion happens on read from smem / write to HBM.
 *
 * Same limitations as the float version: paper-literal loop order,
 * one thread per query row, no tensor cores (plain FFMA32).
 */

template<int Bc=8, int Br=32>
__global__
void flash_attention_v1_kernel(
    const __half* Q,
    const __half* K,
    const __half* V,
    const __half* mask,
    float* l,
    float* m,
    __half* out,
    const int seq_len,
    const int d_model
) {
    assert(Br == blockDim.x);
    const int Tr = (int)ceil((float)seq_len / Br);
    const int Tc = (int)ceil((float)seq_len / Bc);
    constexpr int Bc_padded = ((Bc + 31) / 32) * 32 + 1;

    int tx = threadIdx.x;
    const int d_padded = ((d_model + 31) / 32) * 32 + 1;
    const float scale = 1.0f / sqrtf((float)d_model);

    // Offset into Q,K,V,O,l,m - different for each batch and head
    const int qkv_offset = blockIdx.x * seq_len * d_model;
    const int lm_offset = blockIdx.x * seq_len;
    const int mask_offset = blockIdx.x * seq_len * seq_len;
    Q += qkv_offset; K += qkv_offset; V += qkv_offset; out += qkv_offset;
    l += lm_offset; m += lm_offset;
    mask += mask_offset;

    // SRAM stores __half; compute converts to float on read.
    extern __shared__ __align__(16) char sram_bytes[];
    __half* sram = reinterpret_cast<__half*>(sram_bytes);
    const int tile_size_kv = Bc * d_padded;
    const int tile_size_q = Br * d_padded;
    __half* Qi = sram;
    __half* Kj = &sram[tile_size_q];
    __half* Vj = &sram[tile_size_q + tile_size_kv];

    // S stays FP32 — it's the softmax score / probability buffer.
    __shared__ float S[Br * Bc_padded];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int y = 0; y < Bc; y++) {
            const int kv_row = j * Bc + y;
            for (int k = threadIdx.x; k < d_model; k += blockDim.x) {
                Kj[y * d_padded + k] = kv_row < seq_len ? K[kv_row * d_model + k] : __float2half(0.0f);
                Vj[y * d_padded + k] = kv_row < seq_len ? V[kv_row * d_model + k] : __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)  {
            int q_row = i * Br + tx;
            if (q_row >= seq_len) continue;

            // Load Qi to SRAM
            for (int x = 0; x < d_model; x++) {
                Qi[tx * d_padded + x] = Q[q_row * d_model + x];
            }
            float row_m_prev = m[q_row];
            float row_l_prev = l[q_row];

            // S = QK^T, row_m = rowmax(S). Dot product in FP32.
            float row_m = -INFINITY;
            #pragma unroll
            for (int y = 0; y < Bc; y++) {
                int kv_row = j * Bc + y;
                float sum = 0.0f;
                if (kv_row < seq_len) {
                    for (int x = 0; x < d_model; x++) {
                        sum += __half2float(Qi[tx * d_padded + x])
                             * __half2float(Kj[y * d_padded + x]);
                    }
                }
                sum *= scale;
                if (mask && q_row < seq_len && kv_row < seq_len) {
                    sum += __half2float(mask[q_row * seq_len + kv_row]);
                }
                S[tx * Bc_padded + y] = sum;

                if (sum > row_m) {
                    row_m = sum;
                }
            }

            if (!isfinite(row_m)) continue;

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0.0f;
            #pragma unroll
            for (int y = 0; y < Bc; y++) {
                int kv_row = j * Bc + y;
                if (kv_row < seq_len) {
                    S[tx * Bc_padded + y] = __expf(S[tx * Bc_padded + y] - row_m);
                    row_l += S[tx * Bc_padded + y];
                } else {
                    S[tx * Bc_padded + y] = 0.0f;
                }
            }

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev)
                            + (__expf(row_m - row_m_new) * row_l);

            // Online-softmax epilogue: read prev O in half, combine in float, store half.
            for (int x = 0; x < d_model; x++) {
                float pv = 0.0f;
                #pragma unroll
                for (int y = 0; y < Bc; y++) {
                    int kv_row = j * Bc + y;
                    if (kv_row < seq_len) {
                        pv += S[tx * Bc_padded + y] * __half2float(Vj[y * d_padded + x]);
                    }
                }
                float o_prev = __half2float(out[q_row * d_model + x]);
                float o_new = (1.0f / row_l_new)
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * o_prev)
                     + (__expf(row_m - row_m_new) * pv));
                out[q_row * d_model + x] = __float2half(o_new);
            }
            m[q_row] = row_m_new;
            l[q_row] = row_l_new;
        }
        __syncthreads();
    }
}


__global__ void flash_attention_v1_init_kernel(
    float* l, float* m, __half* out, int n_lm, int n_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_lm) {
        l[idx] = 0.0f;
        m[idx] = -INFINITY;
    }
    if (idx < n_out) {
        out[idx] = __float2half(0.0f);
    }
}


template<int Bc = 8, int Br = 32>
void flash_attention_v1(
    const __half* Q, const __half* K, const __half* V, const __half* mask,
    __half* out, int batch_size, int num_heads, int seq_len, int d_model
) {
    const int bh = batch_size * num_heads;
    const int n_lm = bh * seq_len;
    const int n_out = n_lm * d_model;

    float *l = nullptr, *m = nullptr;
    cudaMalloc(&l, n_lm * sizeof(float));
    cudaMalloc(&m, n_lm * sizeof(float));

    const int init_threads = 256;
    const int init_blocks = (n_out + init_threads - 1) / init_threads;
    flash_attention_v1_init_kernel<<<init_blocks, init_threads>>>(l, m, out, n_lm, n_out);

    const int d_padded = ((d_model + 31) / 32) * 32 + 1;
    const size_t smem_bytes = (Br + 2 * Bc) * d_padded * sizeof(__half);

    dim3 grid(bh);
    dim3 block(Br);
    flash_attention_v1_kernel<Bc, Br><<<grid, block, smem_bytes>>>(
        Q, K, V, mask, l, m, out, seq_len, d_model);

    cudaFree(l);
    cudaFree(m);
}
