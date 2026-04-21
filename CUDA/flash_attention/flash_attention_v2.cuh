#pragma once

#include "../../common/cuda_utils.cuh"
#include "../../common/warp_primitives.cuh"
#include "../block/reduction.cuh"
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// FLASH ATTENTION (FORWARD) - EDUCATIONAL IMPLEMENTATION
// ============================================================================
// This kernel implements the *online softmax* update across J-tiles, avoiding
// materializing the full (seq_len x seq_len) attention matrix.
//
// Notes / limitations (intentional, to keep the implementation simple):
// - float32 only
// - additive mask (0.0 = attend, -inf = mask out) or nullptr
// - currently assumes d_model == JT (default 128) so each thread accumulates
//   exactly one output channel. This matches common head_dim=128.
//
// Shapes:
//   Q, K, V: [batch, heads, seq_len, d_model]
//   mask:    [batch, heads, seq_len, seq_len] (optional)
//   out:     [batch, heads, seq_len, d_model]

#define Q(offset, i, k) Q[(offset) + (i) * d_model + (k)]
#define K(offset, i, k) K[(offset) + (i) * d_model + (k)]
#define V(offset, i, k) V[(offset) + (i) * d_model + (k)]
#define out(offset, i, k) out[(offset) + (i) * d_model + (k)]
#define mask(offset, i, j) mask[(offset_mask) + (i) * seq_len + (j)]
// Pad d_model by 1 to avoid bank conflicts (d_model=128 -> 129 breaks 32-alignment)
// Use runtime d_model_padded instead of macro
#define Q_tiles(i, k) Q_tiles[(i) * d_model_padded + (k)]
#define K_tiles(i, k) K_tiles[(i) * d_model_padded + (k)]
#define V_tiles(i, k) V_tiles[(i) * d_model_padded + (k)]
#define VT_tiles(k, i) VT_tiles[(k) * Bc + (i)]


// ============================================================================
// OPTIMIZED FLASH ATTENTION KERNEL
// ============================================================================
template <typename T, typename AccT=float, int NUM_WARPS_PER_BLOCK = 8, int Br = 8, int Bc = 32, int D_MAX = 4>
__global__ void flash_attention_v2_kernel(
    const T* Q,
    const T* K,
    const T* V,
    const T* mask,
    T* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_model
) {
    assert(Bc % warpSize == 0);
    assert(NUM_WARPS_PER_BLOCK == Br);

    // Calculate d_model_padded at runtime: round up to multiple of 32, then add 1
    const int d_model_padded = ((d_model + 31) / 32) * 32 + 1;

    const int i0 = blockIdx.x * Br;
    const int offset = blockIdx.y * seq_len * d_model;
    const int offset_mask = blockIdx.y * seq_len * seq_len;
    const int lane_id = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int i = warp_id + i0;

    constexpr int warpSize = 32;
    constexpr int ELTS_c = Bc / warpSize;
    AccT O_acc[D_MAX];
    #pragma unroll
    for (int d = 0; d < D_MAX; d++) {
        O_acc[d] = Cast2<AccT, float>::value(0.0f);
    }

    // Shared memory for K and V tiles
    extern __shared__ __align__(16) char smem_bytes[];
    T* smem = reinterpret_cast<T*>(smem_bytes);
    T *Q_tiles = smem;
    T *K_tiles = smem + Br * d_model_padded;
    T *V_tiles = smem + (Bc + Br) * d_model_padded;
    const AccT scale = Cast2<AccT, float>::value(1.0f) / Cast2<AccT, float>::value(sqrtf(d_model));

    // load Q_tile
    for (int k = lane_id; k < d_model; k += warpSize) {
        Q_tiles(warp_id, k) = i < seq_len ? __ldg(&Q(offset, i, k)) : Cast2<T, float>::value(0.0f);
    }
    // __syncthreads();
    
    // each warp doing a single row
    AccT m_i = -Infinity<AccT>::value();
    AccT l_i = Cast2<AccT, float>::value(0.0f);
    // Process each column tile
    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        // Cooperatively load K and V tiles
        for (int j = warp_id; j < Bc; j += NUM_WARPS_PER_BLOCK) {
            for (int k = lane_id; k < d_model; k += warpSize) {
                K_tiles(j, k) = (j0 + j < seq_len) ? __ldg(&K(offset, j0 + j, k)) : Cast2<T, float>::value(0.0f);
                V_tiles(j, k) = (j0 + j < seq_len) ? __ldg(&V(offset, j0 + j, k)) : Cast2<T, float>::value(0.0f);
            }
        }
        __syncthreads();

        // Compute attention scores: Q @ K^T
        AccT S[ELTS_c];
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            S[tc] = Cast2<AccT, float>::value(0.0f);
        }
        
        // Dot product over d_model
        for (int k = 0; k < d_model; k++) {
            const AccT q_ik = Cast2<AccT, T>::value(Q_tiles(warp_id, k));
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                const int j = tc * warpSize + lane_id;
                const AccT k_jk = (j < Bc) ? Cast2<AccT, T>::value(K_tiles(j, k)) : Cast2<AccT, float>::value(0.0f);
                S[tc] += q_ik * k_jk;
            }
        }

        // Scale and apply mask
        AccT local_max_S = -Infinity<AccT>::value();
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            S[tc] *= scale;
            const int j = tc * warpSize + lane_id;
            if (mask && j0 + j < seq_len) {
                S[tc] += Cast2<AccT, T>::value(__ldg(&mask(offset_mask, i, j0 + j)));
            }
            local_max_S = max(local_max_S, S[tc]);
        }
        
        // Warp reduce to find row max
        AccT m_ij = warp_reduce(local_max_S, lane_id, op_max<AccT>(), __activemask());
        m_ij = __shfl_sync(__activemask(), m_ij, 0);

        // Skip if all masked out
        if (!isfinite(Cast2<float, AccT>::value(m_ij))) continue;

        // Compute exp(S - max)
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            S[tc] = ExpTraits<AccT, AccT>::eval(S[tc] - m_ij);
        }

        // Sum for normalization
        AccT local_sum_S = Cast2<AccT, float>::value(0.0f);
        #pragma unroll
        for (int tc = 0; tc < ELTS_c; tc++) {
            local_sum_S += S[tc];
        }
        
        AccT l_ij = warp_reduce(local_sum_S, lane_id, op_sum<AccT>(), __activemask());
        l_ij = __shfl_sync(__activemask(), l_ij, 0);

        // Online softmax statistics update
        const AccT m_i_new = max(m_i, m_ij);
        const AccT alpha = ExpTraits<AccT, AccT>::eval(m_i - m_i_new);
        const AccT beta = ExpTraits<AccT, AccT>::eval(m_ij - m_i_new);
        const AccT l_i_new = alpha * l_i + beta * l_ij;

        // Compute P @ V - optimized with padded layout to avoid bank conflicts
        for (int d = lane_id; d < d_model; d += warpSize) {
            AccT O_id = Cast2<AccT, float>::value(0.0f);
            // Shuffle S values to get all P[i,j] for this d
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                for (int s = 0; s < warpSize; s++) {
                    const AccT P_ij = __shfl_sync(0xffffffff, S[tc], s);
                    const int j = tc * warpSize + s;
                    O_id += P_ij * Cast2<AccT, T>::value(V_tiles(j, d));
                }
            }
            if (i < seq_len) {
                O_acc[d/warpSize] = (l_i * alpha * O_acc[d/warpSize] + beta * O_id) / l_i_new;
            }
        }
        
        // Update statistics for this row
        m_i = m_i_new;
        l_i = l_i_new;
        __syncthreads();
    }

    // write the output
    for (int d = lane_id; d < d_model; d += warpSize) {
        if (i < seq_len) {
            out(offset, i, d) = Cast2<T, AccT>::value(O_acc[d/warpSize]);
        }
    }
}


// ============================================================================
// BASELINE FLASH ATTENTION KERNEL (original implementation for comparison)
// ============================================================================
template <typename T, typename AccT=float, int NUM_WARPS_PER_BLOCK = 4, int Br = 64, int Bc = 64>
__global__ void flash_attention_v2_baseline_kernel(
    const T* Q,
    const T* K,
    const T* V,
    const T* mask,
    T* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_model
) {
    assert(Bc % warpSize == 0);
    assert(d_model % warpSize == 0);

    const int i0 = blockIdx.x * Br;
    const int offset = blockIdx.y * seq_len * d_model;
    const int offset_mask = blockIdx.y * seq_len * seq_len;
    const int lane_id = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    constexpr int warpSize = 32;
    constexpr int ELTS_c = Bc / warpSize;  // Assume Bc is multiple of warpSize
    // constexpr int ELTS_r = Br / NUM_WARPS_PER_BLOCK;  // Assume Br is multiple of warpSize

    // Pad d_model by 1 to avoid bank conflicts (same as optimized kernel)
    const int d_model_padded = d_model + 1;

    // shared memory for K and V tiles
    extern __shared__ __align__(16) char smem_bytes[];
    T* smem = reinterpret_cast<T*>(smem_bytes);
    T *K_tiles = smem;
    T *V_tiles = smem + Bc * d_model_padded;
    __shared__ AccT m_i_tiles[Br];
    __shared__ AccT l_i_tiles[Br];
    
    // initialize m_i and l_i
    for (int i = threadIdx.x; i < Br; i += blockDim.x) {
        m_i_tiles[i] = -Infinity<AccT>::value();
        l_i_tiles[i] = Cast2<AccT, float>::value(0.0f);
    }
    __syncthreads();
    const AccT scale = Cast2<AccT, float>::value(1.0f) / Cast2<AccT, float>::value(sqrt((float)d_model));

    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        // load tile of K and V
        for (int j = warp_id; j < Bc; j += NUM_WARPS_PER_BLOCK) {
            for (int k = lane_id; k < d_model; k += warpSize) {
                K_tiles(j, k) = j0 + j < seq_len ? __ldg(&K(offset, j0 + j, k)) : Cast2<T, float>::value(0.0f);
                V_tiles(j, k) = j0 + j < seq_len ? __ldg(&V(offset, j0 + j, k)) : Cast2<T, float>::value(0.0f);
            }
        }
        __syncthreads();
        
        for (int i = warp_id; i < Br; i += NUM_WARPS_PER_BLOCK) {
            // option1: 1D subtile of S, i.e. warp-per-row
            AccT S[ELTS_c];
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                S[tc] = Cast2<AccT, float>::value(0.0f);
            }
            // compute scores S = Q @ K^T, Br x Bc
            for (int k = 0; k < d_model; k++) {
                const AccT q_ik = i0 + i < seq_len ? Cast2<AccT, T>::value(Q(offset, i0 + i, k)) : Cast2<AccT, float>::value(0.0f);
                #pragma unroll
                for (int tc = 0; tc < ELTS_c; tc++) {
                    const int j = tc * warpSize + lane_id;
                    const AccT k_jk = j < Bc ? Cast2<AccT, T>::value(K_tiles(j, k)) : Cast2<AccT, float>::value(0.0f);
                    S[tc] += q_ik * k_jk;
                }
            }

            // find max value in the row
            AccT local_max_S = -Infinity<AccT>::value();
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                S[tc] *= scale;
                const int j = tc * warpSize + lane_id;
                if (mask && j0 + j < seq_len && i0 + i < seq_len) {
                    S[tc] += Cast2<AccT, T>::value(__ldg(&mask(offset_mask, i0 + i, j0 + j)));
                }
                local_max_S = max(local_max_S, S[tc]);
            }
            
            AccT m_ij = warp_reduce(local_max_S, lane_id, op_max<AccT>(), __activemask());
            m_ij = __shfl_sync(__activemask(), m_ij, 0); // broadcast the max value to all lanes

            // if all masked out, skip the update
            if (!isfinite(Cast2<float, AccT>::value(m_ij))) continue;

            // apply softmax, S now store P = exp(S - max_S)
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                S[tc] = ExpTraits<AccT, AccT>::eval(S[tc] - m_ij);
            }

            // reduce the row
            AccT local_sum_S = Cast2<AccT, float>::value(0.0f);
            #pragma unroll
            for (int tc = 0; tc < ELTS_c; tc++) {
                local_sum_S += S[tc];
            }
            
            AccT l_ij = warp_reduce(local_sum_S, lane_id, op_sum<AccT>(), __activemask());
            l_ij = __shfl_sync(__activemask(), l_ij, 0); // broadcast the sum value to all lanes

            // update the accumulator
            const AccT m_i = m_i_tiles[i];
            const AccT l_i = l_i_tiles[i];

            const AccT m_i_new = max(m_i, m_ij);
            const AccT alpha = ExpTraits<AccT, AccT>::eval(m_i - m_i_new);
            const AccT beta = ExpTraits<AccT, AccT>::eval(m_ij - m_i_new);
            const AccT l_i_new = alpha * l_i + beta * l_ij;

            // update the output 
            // calculate P@V
            // paralell over d, col of output lanes 
            for (int d = lane_id; d < d_model; d += warpSize) {
                AccT O_id = Cast2<AccT, float>::value(0.0f);
                for (int s = 0; s < warpSize; s ++) {
                    for (int tc = 0; tc < ELTS_c; tc ++) {
                        // broadcast value of S[t]
                        const AccT P_ij = __shfl_sync(0xffffffff, S[tc], s);
                        const int j = tc * warpSize + s;
                        if (j0 + j < seq_len) {
                            O_id += P_ij * Cast2<AccT, T>::value(V_tiles(j, d));
                        }
                    }
                }
                if (i0 + i < seq_len) {
                    out(offset, i0 + i, d) = Cast2<T, AccT>::value((l_i * alpha * Cast2<AccT, T>::value(out(offset, i0 + i, d)) + beta * O_id) / l_i_new);
                }
            }
            
            // update the accumulator
            if (lane_id == 0) {
                m_i_tiles[i] = m_i_new;
                l_i_tiles[i] = l_i_new;
            }
        }
        __syncthreads();
    }
}


#undef Q
#undef K
#undef V
#undef out
#undef mask
#undef Q_tiles
#undef K_tiles
#undef V_tiles
#undef VT_tiles