#pragma once
#include <cute/tensor.hpp>

/*
 * SMEM SGEMM using CuTe: C = alpha * A * B + beta * C
 * A(M,K), B(K,N), C(M,N), all column-major, single precision.
 *
 * BLK_M x BLK_N threads per CTA, K=1 tile (no K-blocking).
 * shared memory tiling.
 */

template <class ProblemShape, class CtaTiler,
    class AStride, class AThreadLayout, class ASmemLayout, 
    class BStride, class BThreadLayout, class BSmemLayout, 
    class CStride, class CThreadLayout, class CSmemLayout>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void sgemm_smem_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    float alpha,
    const float* A, AStride dA, AThreadLayout tA, ASmemLayout sA_layout, 
    const float* B, BStride dB, BThreadLayout tB, BSmemLayout sB_layout, 
    float beta,
    float*       C, CStride dC, CThreadLayout tC, CSmemLayout)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // M x K
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // N x K
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // M x N

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    // create cta tile
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // BLK_M x BLK_K x k
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // BLK_N x BLK_K x k
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // BLK_M x BLK_N

    // allocate shared memory and create tensor 
    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // BLK_M x BLK_K
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // BLK_N x BLK_K

    // calculate thread's local partition
    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // 1 x 1 x k
    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // 1 x 1 x k
    Tensor tCgC = local_partition(gC, tC, threadIdx.x); // 1 x 1 

    // tensor for loading smem 
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // 1 x 1
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // 1 x 1

    // view tensor for calculation (smem partitioned by C thread layout)
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // 1 x BLK_K
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // 1 x BLK_K

    // register fragments for A and B (smem → register before gemm)
    Tensor tCrA = make_tensor_like(tCsA); // 1 x BLK_K in registers
    Tensor tCrB = make_tensor_like(tCsB); // 1 x BLK_K in registers

    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile += 1) {
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);
        __syncthreads();
        copy(tCsA, tCrA); // smem → register
        copy(tCsB, tCrB); // smem → register
        // cute::gemm is much faster on register than smem
        gemm(tCrA, tCrB, tCrC);
        __syncthreads();
    }

    // Epilogue in-place: C = alpha * acc + beta * C.
    axpby(alpha, tCrC, beta, tCgC);
}

// SGEMM: C = alpha * A * B + beta * C  (float32 in/out)
// A(M,K), B(K,N), C(M,N), all column-major
template <int BLOCK_SIZE = 32>
void sgemm_smem(
    int m, int n, int k,
    float alpha,
    const float* A, int ldA,
    const float* B, int ldB,
    float beta,
    float* C, int ldC)
{
    using namespace cute;

    auto cta_tiler = make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{});
    auto shape_MNK = make_shape(m, n, k);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // for coalesced access
    auto tA = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}));
    auto tB = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}), LayoutRight{});
    auto tC = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}));

    // define smem layout (column major)
    auto sA_layout = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}));
    auto sB_layout = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}), LayoutRight{});
    auto sC_layout = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}));

    dim3 block_size(size(tC));
    dim3 grid_size(size(ceil_div(m, Int<BLOCK_SIZE>{})),
                   size(ceil_div(n, Int<BLOCK_SIZE>{})));

    sgemm_smem_device<<<grid_size, block_size>>>(
        shape_MNK, cta_tiler,
        alpha,
        A, dA, tA, sA_layout, 
        B, dB, tB, sB_layout, 
        beta,
        C, dC, tC, sC_layout);
}
