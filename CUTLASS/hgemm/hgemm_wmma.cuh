#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>

/*
 * Warp-level tensor-core HGEMM using CuTe: C = alpha * A^T * B + beta * C   (TN layout)
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * Uses SM80_16x8x16_F32F16F16F32_TN (mma.sync, warp-level, synchronous).
 * Swizzled smem + ldmatrix s2r: warp-cooperative 8x8 half-tile loads,
 * bank-conflict free. No double buffering.
 */

template <class ProblemShape, class CtaTiler,
    class AStride, class TiledCopyA, class S2RCopyA, class ASmemLayout,
    class BStride, class TiledCopyB, class S2RCopyB, class BSmemLayout,
    class CStride, class TiledMMA>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMMA{}))::value)
void hgemm_wmma_device(
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

    // shared memory — use layout_b() for composed (swizzled) layouts
    __shared__ cute::half_t smemA[cosize_v<decltype(sA_layout.layout_b())>];
    __shared__ cute::half_t smemB[cosize_v<decltype(sB_layout.layout_b())>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // gmem → smem partitions
    auto thr_g2s_a = g2s_A.get_slice(threadIdx.x);
    Tensor tAgA = thr_g2s_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)
    Tensor tAsA = thr_g2s_a.partition_D(sA);   // (CPY, CPY_M, CPY_K)

    auto thr_g2s_b = g2s_B.get_slice(threadIdx.x);
    Tensor tBgB = thr_g2s_b.partition_S(gB);
    Tensor tBsB = thr_g2s_b.partition_D(sB);

    // MMA partitions
    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                     // (MMA, MMA_M, MMA_N)
    Tensor tCrA = thr_mma.partition_fragment_A(sA);            // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);            // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);               // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // smem → register partitions
    auto s2r_thr_a = s2r_A.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_a.partition_S(sA);                   // (CPY, MMA_M, MMA_K)
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);                    // (CPY, MMA_M, MMA_K)

    auto s2r_thr_b = s2r_B.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_b.partition_S(sB);                   // (CPY, MMA_N, MMA_K)
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);                    // (CPY, MMA_N, MMA_K)

    auto K_TILE_MAX  = size<3>(tAgA);   // number of BK tiles along K
    auto K_BLOCK_MAX = size<2>(tCrA);   // inner MMA-K iterations per BK tile

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // gmem → smem
        copy(g2s_A, tAgA(_, _, _, k_tile), tAsA);
        copy(g2s_B, tBgB(_, _, _, k_tile), tBsB);
        __syncthreads();

        // smem → register, then MMA
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            copy(s2r_A, tXsA(_, _, k_block), tXrA(_, _, k_block));
            copy(s2r_B, tXsB(_, _, k_block), tXrB(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
        __syncthreads();
    }

    // epilogue: C = alpha * acc + beta * C
    axpby(alpha, tCrC, beta, tCgC);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32>
void hgemm_wmma(
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
    // Per-step coverage: 32 in M, 16 in N
    // ---------------------------------------------------------------
    auto mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<_2, _2, _1>>{}     // (warpM, warpN, warpK)
    );

    constexpr int NUM_THREADS = size(decltype(mma){});   // 128

    // ---------------------------------------------------------------
    // Shared memory: swizzled LayoutRight (K-contiguous)
    //
    // Swizzle<B, M, S> XORs bits [S+B-1:S] into bits [M+B-1:M].
    // M=3 preserves 8-element (128-bit) vectorized access alignment.
    //
    // Focus on S2Rcopy bank conflict which is critical
    // sA atom (16, BK): MMA reads M={0,2,4,6} — m[2:1] at bits [7:6]
    //   Swizzle<2,3,6> → XOR bits [7:6] into [4:3] → 0 conflicts
    //
    // sB atom (8, BK):  MMA reads N={0..7} — N[2] at bit [6] aliases mod 32
    //   Swizzle<2,3,6> → XOR bits [7:6] into [4:3] → 0 conflicts
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
    // gmem → smem: 128-bit vectorized copies
    // ---------------------------------------------------------------
    constexpr int VEC    = 8;            // 8 x half = 128 bits
    constexpr int BK_VEC = BK / VEC;

    using G2SCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, cute::half_t>;
    
    // G2S have 4-way bank conflict that can't be eliminated
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
    // copyA need to move 16x16 half, need U32x4_N, each thread read contiguous 128 bits
    // copyB need to move 16x8 half, only need U32x2_N
    // both smem and register k-contiguous thus no transposed needed.
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>;
    auto s2r_copy_A = make_tiled_copy_A(S2RCopyAtomA{}, mma);
    auto s2r_copy_B = make_tiled_copy_B(S2RCopyAtomB{}, mma);

    // ---------------------------------------------------------------
    // Launch
    // ---------------------------------------------------------------
    dim3 block(NUM_THREADS);
    dim3 grid(size(ceil_div(m, Int<BM>{})),
              size(ceil_div(n, Int<BN>{})));

    hgemm_wmma_device<<<grid, block>>>(
        shape_MNK, cta_tiler, alpha,
        A, dA, g2s_copy_A, s2r_copy_A, sA_layout,
        B, dB, g2s_copy_B, s2r_copy_B, sB_layout,
        beta, C, dC, mma);
}
