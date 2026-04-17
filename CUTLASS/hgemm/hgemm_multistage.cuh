#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm80.hpp>

/*
 * Multistage tensor-core HGEMM: C = alpha * A^T * B + beta * C   (TN layout)
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * Uses SM80_16x8x16_F32F16F16F32_TN with cp.async pipelined gmem→smem.
 * NUM_STAGES smem buffers in a circular pipeline:
 *   prologue : fill stages 0 .. NUM_STAGES-2
 *   mainloop : compute oldest ready stage, issue next load
 *   epilogue : write C
 *
 * NUM_STAGES=2 is classic double buffering.
 */

template <int NUM_STAGES,
    class ProblemShape, class CtaTiler,
    class AStride, class TiledCopyA, class S2RCopyA, class ASmemLayout,
    class BStride, class TiledCopyB, class S2RCopyB, class BSmemLayout,
    class CStride, class TiledMMA>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMMA{}))::value)
void hgemm_multistage_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    float alpha,
    const cute::half_t* A, AStride dA, TiledCopyA g2s_A, S2RCopyA s2r_A, ASmemLayout sA_layout,
    const cute::half_t* B, BStride dB, TiledCopyB g2s_B, S2RCopyB s2r_B, BSmemLayout sB_layout,
    float beta,
    cute::half_t*       C, CStride dC, TiledMMA mma)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BM, BK, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BN, BK, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BM, BN)

    // Multistage shared memory via dynamic smem
    constexpr int smemA_elem = cosize_v<decltype(sA_layout.layout_b())>;
    constexpr int smemB_elem = cosize_v<decltype(sB_layout.layout_b())>;
    extern __shared__ __align__(128) cute::half_t smem_buf[];
    cute::half_t* smemA = smem_buf;
    cute::half_t* smemB = smem_buf + NUM_STAGES * smemA_elem;

    // gmem → smem partitions (source side only; dest rebuilt per stage)
    auto thr_g2s_a = g2s_A.get_slice(threadIdx.x);
    Tensor tAgA = thr_g2s_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)

    auto thr_g2s_b = g2s_B.get_slice(threadIdx.x);
    Tensor tBgB = thr_g2s_b.partition_S(gB);

    // MMA partitions (use stage-0 smem as shape reference)
    Tensor sA_ref = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB_ref = make_tensor(make_smem_ptr(smemB), sB_layout);

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.partition_fragment_A(sA_ref);
    Tensor tCrB = thr_mma.partition_fragment_B(sB_ref);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // smem → register retile (shape-only, partition_S rebuilt per stage)
    auto s2r_thr_a = s2r_A.get_slice(threadIdx.x);
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);

    auto s2r_thr_b = s2r_B.get_slice(threadIdx.x);
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);

    auto K_TILE_MAX  = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    // Helper: issue async copy for tile k_tile into stage
    auto issue_load = [&](int k_tile, int stage) {
        Tensor sA_w = make_tensor(make_smem_ptr(smemA + stage * smemA_elem), sA_layout);
        Tensor sB_w = make_tensor(make_smem_ptr(smemB + stage * smemB_elem), sB_layout);
        copy(g2s_A, tAgA(_, _, _, k_tile), thr_g2s_a.partition_D(sA_w));
        copy(g2s_B, tBgB(_, _, _, k_tile), thr_g2s_b.partition_D(sB_w));
        cp_async_fence();
    };

    // Helper: compute MMA for a given stage
    auto compute_stage = [&](int stage) {
        Tensor sA_r = make_tensor(make_smem_ptr(smemA + stage * smemA_elem), sA_layout);
        Tensor sB_r = make_tensor(make_smem_ptr(smemB + stage * smemB_elem), sB_layout);
        Tensor tXsA = s2r_thr_a.partition_S(sA_r);
        Tensor tXsB = s2r_thr_b.partition_S(sB_r);
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            copy(s2r_A, tXsA(_, _, k_block), tXrA(_, _, k_block));
            copy(s2r_B, tXsB(_, _, k_block), tXrB(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    };

    // ========== Prologue: fill stages 0 .. NUM_STAGES-2 ==========
    CUTE_UNROLL
    for (int s = 0; s < NUM_STAGES - 1 && s < K_TILE_MAX; ++s) {
        issue_load(s, s);
    }

    // ========== Mainloop ==========
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // Wait for this tile's async copy to land
        cp_async_wait<NUM_STAGES - 2>();
        __syncthreads();

        // Compute on the ready stage
        compute_stage(k_tile % NUM_STAGES);

        // Issue next load into the stage we just freed
        int next_tile = k_tile + (NUM_STAGES - 1);
        if (next_tile < K_TILE_MAX) {
            issue_load(next_tile, next_tile % NUM_STAGES);
        }

        __syncthreads();
    }

    // ========== Epilogue ==========
    axpby(alpha, tCrC, beta, tCgC);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32, int NUM_STAGES = 3>
void hgemm_multistage(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto shape_MNK = make_shape(m, n, k);

    // TN layout strides
    auto dA = make_stride(ldA, Int<1>{});   // A(M,K) K-contiguous
    auto dB = make_stride(ldB, Int<1>{});   // B(N,K) K-contiguous
    auto dC = make_stride(Int<1>{}, ldC);   // C(M,N) M-contiguous (col-major)

    // ---------------------------------------------------------------
    // Tensor-core MMA: 16x8x16 with F32 accumulator
    // Tile 2 warps in M, 2 in N → 4 warps = 128 threads
    // ---------------------------------------------------------------
    auto mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<_2, _2, _1>>{}
    );

    constexpr int NUM_THREADS = size(decltype(mma){});   // 128

    // ---------------------------------------------------------------
    // Shared memory: swizzled LayoutRight (K-contiguous)
    // Same swizzle as hgemm_wmma.cuh
    // ---------------------------------------------------------------
    auto sA_layout = tile_to_shape(
        composition(Swizzle<2, 3, 6>{},
            make_layout(make_shape(Int<16>{}, Int<BK>{}), LayoutRight{})),
        make_shape(Int<BM>{}, Int<BK>{}));
    auto sB_layout = tile_to_shape(
        composition(Swizzle<2, 3, 6>{},
            make_layout(make_shape(Int<8>{}, Int<BK>{}), LayoutRight{})),
        make_shape(Int<BN>{}, Int<BK>{}));

    // ---------------------------------------------------------------
    // gmem → smem: cp.async 128-bit copies (bypass L1)
    // ---------------------------------------------------------------
    constexpr int VEC    = 8;            // 8 x half = 128 bits
    constexpr int BK_VEC = BK / VEC;

    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>;

    constexpr int ThrM_A = NUM_THREADS / BK_VEC;
    auto g2s_copy_A = make_tiled_copy(G2SCopyAtom{},
        make_layout(make_shape(Int<ThrM_A>{}, Int<BK_VEC>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{},      Int<VEC>{})));

    constexpr int ThrN_B = NUM_THREADS / BK_VEC;
    auto g2s_copy_B = make_tiled_copy(G2SCopyAtom{},
        make_layout(make_shape(Int<ThrN_B>{}, Int<BK_VEC>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{},      Int<VEC>{})));

    // ---------------------------------------------------------------
    // smem → register: scalar copies, retiled to match MMA layout
    // ---------------------------------------------------------------
    // ldmatrix: warp-cooperative 8x8 half tile loads, bank-conflict free
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>;
    auto s2r_copy_A = make_tiled_copy_A(S2RCopyAtomA{}, mma);
    auto s2r_copy_B = make_tiled_copy_B(S2RCopyAtomB{}, mma);

    // ---------------------------------------------------------------
    // Launch with dynamic shared memory
    // ---------------------------------------------------------------
    constexpr int smemA_elem = cosize_v<decltype(sA_layout.layout_b())>;
    constexpr int smemB_elem = cosize_v<decltype(sB_layout.layout_b())>;
    constexpr int smem_bytes = NUM_STAGES * (smemA_elem + smemB_elem) * sizeof(cute::half_t);

    dim3 block(NUM_THREADS);
    dim3 grid(size(ceil_div(m, Int<BM>{})),
              size(ceil_div(n, Int<BN>{})));

    auto kernel = hgemm_multistage_device<NUM_STAGES,
        decltype(shape_MNK), decltype(cta_tiler),
        decltype(dA), decltype(g2s_copy_A), decltype(s2r_copy_A), decltype(sA_layout),
        decltype(dB), decltype(g2s_copy_B), decltype(s2r_copy_B), decltype(sB_layout),
        decltype(dC), decltype(mma)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    kernel<<<grid, block, smem_bytes>>>(
        shape_MNK, cta_tiler, alpha,
        A, dA, g2s_copy_A, s2r_copy_A, sA_layout,
        B, dB, g2s_copy_B, s2r_copy_B, sB_layout,
        beta, C, dC, mma);
}
