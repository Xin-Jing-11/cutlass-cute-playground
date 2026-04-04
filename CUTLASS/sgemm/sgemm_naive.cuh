#pragma once
#include <cute/tensor.hpp>

/*
 * Naive SGEMM using CuTe: C = alpha * A * B + beta * C
 * A(M,K), B(K,N), C(M,N), all column-major, single precision.
 *
 * BLK_M x BLK_N threads per CTA, K=1 tile (no K-blocking).
 * No shared memory, no register tiling.
 */

template <class ProblemShape, class CtaTiler,
    class AStride, class AThreadLayout,
    class BStride, class BThreadLayout,
    class CStride, class CThreadLayout>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void sgemm_naive_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    float alpha,
    const float* A, AStride dA, AThreadLayout,
    const float* B, BStride dB, BThreadLayout,
    float beta,
    float*       C, CStride dC, CThreadLayout tC)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // M x K
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // N x K
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // M x N

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // BLK_M x BLK_K x k
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // BLK_N x BLK_K x k
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // BLK_M x BLK_N

    Tensor tCgA = local_partition(gA, tC, threadIdx.x, Step<_1, X>{}); // THR_M x BLK_K x k
    Tensor tCgB = local_partition(gB, tC, threadIdx.x, Step<X, _1>{}); // THR_N x BLK_K x k
    Tensor tCgC = local_partition(gC, tC, threadIdx.x);                // THR_M x THR_N
    Tensor tCrA = make_tensor_like(tCgA(_, _, 0)); // THR_M x BLK_K
    Tensor tCrB = make_tensor_like(tCgB(_, _, 0)); // THR_N x BLK_K

    // Float32 accumulator
    Tensor tCrC = make_tensor<float>(shape(tCgC));
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tCgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile += 1) {
        copy(tCgA(_, _, k_tile), tCrA);
        copy(tCgB(_, _, k_tile), tCrB);
        gemm(tCrA, tCrB, tCrC);
    }

    // Epilogue in-place: C = alpha * acc + beta * C.
    axpby(alpha, tCrC, beta, tCgC);
}

// SGEMM: C = alpha * A * B + beta * C  (float32 in/out)
// A(M,K), B(K,N), C(M,N), all column-major
template <int BLOCK_SIZE = 32>
void sgemm_naive(
    int m, int n, int k,
    float alpha,
    const float* A, int ldA,
    const float* B, int ldB,
    float beta,
    float* C, int ldC)
{
    using namespace cute;

    auto cta_tiler = make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}, Int<1>{});
    auto shape_MNK = make_shape(m, n, k);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // for coalesced access
    auto tA = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}));
    auto tB = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}), LayoutRight{});
    auto tC = make_layout(make_shape(Int<BLOCK_SIZE>{}, Int<BLOCK_SIZE>{}));

    dim3 block_size(size(tC));
    dim3 grid_size(size(ceil_div(m, Int<BLOCK_SIZE>{})),
                   size(ceil_div(n, Int<BLOCK_SIZE>{})));

    sgemm_naive_device<<<grid_size, block_size>>>(
        shape_MNK, cta_tiler,
        alpha,
        A, dA, tA,
        B, dB, tB,
        beta,
        C, dC, tC);
}
