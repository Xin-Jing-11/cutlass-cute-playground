#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Flash Attention v1 (CUTLASS/CuTe, no tensor cores — UniversalFMA).
//
// Paper form: j outer, i inner; 1 thread = 1 query row within a CTA.
// l, m, O live in gmem and are read-modify-written per j-tile.
// Layout: [batch, heads, seq_len, d_model] row-major, cute::half_t storage, float compute.
//
// Grid : batch * heads                   (one CTA per head)
// Block: Br threads

template <int Bc, int Br>
__global__ static void flash_attention_naive_device(
    int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask,
    float* l, float* m, cute::half_t* out)
{
    using namespace cute;

    const int bh       = blockIdx.x;
    const int qkv_off  = bh * seq_len * d_model;
    const int lm_off   = bh * seq_len;
    const int mask_off = bh * seq_len * seq_len;

    const int tx     = threadIdx.x;
    const int Tr     = (seq_len + Br - 1) / Br;
    const int Tc     = (seq_len + Bc - 1) / Bc;
    const int d_pad  = ((d_model + 31) / 32) * 32 + 1;
    const float scale = 1.0f / sqrtf((float)d_model);

    Tensor mQ = make_tensor(make_gmem_ptr(Q + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));
    Tensor mK = make_tensor(make_gmem_ptr(K + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));
    Tensor mV = make_tensor(make_gmem_ptr(V + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));
    Tensor mO = make_tensor(make_gmem_ptr(out + qkv_off),
                            make_shape(seq_len, d_model), make_stride(d_model, 1));

    extern __shared__ __align__(16) char smem_raw[];
    cute::half_t* smem = reinterpret_cast<cute::half_t*>(smem_raw);
    cute::half_t* sQ_p = smem;
    cute::half_t* sK_p = sQ_p + Br * d_pad;
    cute::half_t* sV_p = sK_p + Bc * d_pad;

    Tensor sQ = make_tensor(make_smem_ptr(sQ_p),
                            make_layout(make_shape(Br, d_model), make_stride(d_pad, 1)));
    Tensor sK = make_tensor(make_smem_ptr(sK_p),
                            make_layout(make_shape(Bc, d_model), make_stride(d_pad, 1)));
    Tensor sV = make_tensor(make_smem_ptr(sV_p),
                            make_layout(make_shape(Bc, d_model), make_stride(d_pad, 1)));

    [[maybe_unused]] auto tiled_mma = make_tiled_mma(
        cute::MMA_Atom<cute::UniversalFMA<float, cute::half_t, cute::half_t, float>>{},
        make_layout(make_shape(Int<Br>{}, Int<1>{})));

    float S_reg[Bc];  // per-thread row of scores

    for (int j = 0; j < Tc; j++) {
        // Load Kj, Vj tiles (cooperative across Br threads).
        for (int y = 0; y < Bc; y++) {
            int kv_row = j * Bc + y;
            for (int k = tx; k < d_model; k += Br) {
                sK(y, k) = (kv_row < seq_len) ? mK(kv_row, k) : cute::half_t(0.0f);
                sV(y, k) = (kv_row < seq_len) ? mV(kv_row, k) : cute::half_t(0.0f);
            }
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            int q_row = i * Br + tx;
            if (q_row >= seq_len) continue;

            // Load one Q row into smem (for bank-friendly reuse across the Bc loop).
            for (int x = 0; x < d_model; x++) sQ(tx, x) = mQ(q_row, x);

            float row_m_prev = m[lm_off + q_row];
            float row_l_prev = l[lm_off + q_row];

            // S = Q_i K_j^T, then row max.
            float row_m = -INFINITY;
            #pragma unroll
            for (int y = 0; y < Bc; y++) {
                int kv_row = j * Bc + y;
                float sum = 0.0f;
                if (kv_row < seq_len) {
                    for (int x = 0; x < d_model; x++) {
                        sum += float(sQ(tx, x)) * float(sK(y, x));
                    }
                }
                sum *= scale;
                if (mask && kv_row < seq_len) {
                    sum += float(mask[mask_off + q_row * seq_len + kv_row]);
                }
                S_reg[y] = sum;
                if (sum > row_m) row_m = sum;
            }
            if (!isfinite(row_m)) continue;

            // P = exp(S - row_m), row_l = sum P.
            float row_l = 0.0f;
            #pragma unroll
            for (int y = 0; y < Bc; y++) {
                int kv_row = j * Bc + y;
                if (kv_row < seq_len) {
                    S_reg[y] = __expf(S_reg[y] - row_m);
                    row_l   += S_reg[y];
                } else {
                    S_reg[y] = 0.0f;
                }
            }

            const float row_m_new = fmaxf(row_m_prev, row_m);
            const float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev
                                  + __expf(row_m      - row_m_new) * row_l;

            // O ← (l_prev α O_prev + β P V) / l_new
            for (int x = 0; x < d_model; x++) {
                float pv = 0.0f;
                #pragma unroll
                for (int y = 0; y < Bc; y++) {
                    int kv_row = j * Bc + y;
                    if (kv_row < seq_len) pv += S_reg[y] * float(sV(y, x));
                }
                float o_prev = float(mO(q_row, x));
                float o_new  = (__expf(row_m_prev - row_m_new) * row_l_prev * o_prev
                              + __expf(row_m      - row_m_new) * pv) / row_l_new;
                mO(q_row, x) = cute::half_t(o_new);
            }
            m[lm_off + q_row] = row_m_new;
            l[lm_off + q_row] = row_l_new;
        }
        __syncthreads();
    }
}


__global__ static void flash_attention_naive_init_kernel(
    float* l, float* m, cute::half_t* out, int n_lm, int n_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_lm) {
        l[idx] = 0.0f;
        m[idx] = -INFINITY;
    }
    if (idx < n_out) out[idx] = cute::half_t(0.0f);
}


template <int Bc = 8, int Br = 32>
void flash_attention_naive(
    int batch_size, int num_heads, int seq_len, int d_model,
    const cute::half_t* Q, const cute::half_t* K, const cute::half_t* V,
    const cute::half_t* mask, cute::half_t* out)
{
    const int bh    = batch_size * num_heads;
    const int n_lm  = bh * seq_len;
    const int n_out = n_lm * d_model;

    float *l = nullptr, *m = nullptr;
    cudaMalloc(&l, n_lm * sizeof(float));
    cudaMalloc(&m, n_lm * sizeof(float));

    const int init_threads = 256;
    const int init_blocks  = (n_out + init_threads - 1) / init_threads;
    flash_attention_naive_init_kernel<<<init_blocks, init_threads>>>(l, m, out, n_lm, n_out);

    const int d_pad = ((d_model + 31) / 32) * 32 + 1;
    const size_t smem = (Br + 2 * Bc) * d_pad * sizeof(cute::half_t);

    dim3 grid(bh);
    dim3 block(Br);
    flash_attention_naive_device<Bc, Br><<<grid, block, smem>>>(
        seq_len, d_model, Q, K, V, mask, l, m, out);

    cudaFree(l);
    cudaFree(m);
}
