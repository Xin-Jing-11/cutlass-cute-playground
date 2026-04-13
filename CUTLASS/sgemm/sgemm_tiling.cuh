#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>

/*
 * Tiling SGEMM using CuTe: D = alpha * A^T * B + beta * C   (TN layout)
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * Full CuTe pattern matching sgemm_sm80.cu:
 *   - tiled_copy for gmem→smem (partition_D handles swizzle)
 *   - tiled_mma for compute (partition_fragment_A/B)
 *   - make_tiled_copy_A/B for smem→register (partition_S handles swizzle)
 *   - Swizzle in ComposedLayout via tile_to_shape
 */

template <class ProblemShape, class CtaTiler,
    class AStride, class TiledCopyA, class S2RCopyA, class ASmemLayout,
    class BStride, class TiledCopyB, class S2RCopyB, class BSmemLayout,
    class CStride, class TiledMMA, class CSmemLayout>
__global__ static
__launch_bounds__(decltype(size(TiledMMA{}))::value)
void sgemm_tiling_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    float alpha,
    const float* A, AStride dA, TiledCopyA g2s_A, S2RCopyA s2r_A, ASmemLayout sA_layout,
    const float* B, BStride dB, TiledCopyB g2s_B, S2RCopyB s2r_B, BSmemLayout sB_layout,
    float beta,
    float*       C, CStride dC, TiledMMA mma, CSmemLayout)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // BM x BK x k
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // BN x BK x k
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // BM x BN

    // allocate smem — cosize from the outer (plain) layout of the composition
    __shared__ __align__(128) float smemA[cosize_v<decltype(sA_layout.layout_b())>];
    __shared__ __align__(128) float smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // swizzled ComposedLayout
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // swizzled ComposedLayout

    // --- gmem→smem via tiled_copy ---
    auto thr_g2s_a = g2s_A.get_slice(threadIdx.x);
    Tensor tAgA = thr_g2s_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)
    Tensor tAsA = thr_g2s_a.partition_D(sA);   // (CPY, CPY_M, CPY_K) — swizzle-aware

    auto thr_g2s_b = g2s_B.get_slice(threadIdx.x);
    Tensor tBgB = thr_g2s_b.partition_S(gB);
    Tensor tBsB = thr_g2s_b.partition_D(sB);

    // --- MMA partitions ---
    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                     // (MMA, MMA_M, MMA_N)
    Tensor tCrA = thr_mma.partition_fragment_A(sA);            // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);            // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);               // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // --- smem→register via make_tiled_copy_A/B (swizzle-aware partition_S) ---
    auto s2r_thr_a = s2r_A.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_a.partition_S(sA);                   // (CPY, MMA_M, MMA_K)
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);                    // (CPY, MMA_M, MMA_K)

    auto s2r_thr_b = s2r_B.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_b.partition_S(sB);                   // (CPY, MMA_N, MMA_K)
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);                    // (CPY, MMA_N, MMA_K)

#ifdef DEBUG
    if (thread0()) {
        print("  sA   : "); print(  sA.layout()); print("\n");
        print("  sB   : "); print(  sB.layout()); print("\n");
        print("tAgA   : "); print(tAgA.layout()); print("\n");
        print("tAsA   : "); print(tAsA.layout()); print("\n");
        print("tCgC   : "); print(tCgC.layout()); print("\n");
        print("tCrA   : "); print(tCrA.layout()); print("\n");
        print("tCrB   : "); print(tCrB.layout()); print("\n");
        print("tCrC   : "); print(tCrC.layout()); print("\n");
        print("tXsA   : "); print(tXsA.layout()); print("\n");
        print("tXrA   : "); print(tXrA.layout()); print("\n");
        print("tXsB   : "); print(tXsB.layout()); print("\n");
        print("tXrB   : "); print(tXrB.layout()); print("\n");
    }
#endif

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile += 1) {
        // gmem → smem
        copy(g2s_A, tAgA(_, _, _, k_tile), tAsA);
        copy(g2s_B, tBgB(_, _, _, k_tile), tBsB);
        __syncthreads();
        // inner K-loop: smem→register, then gemm
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            copy(s2r_A, tXsA(_, _, k_block), tXrA(_, _, k_block));
            copy(s2r_B, tXsB(_, _, k_block), tXrB(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
        __syncthreads();
    }

    axpby(alpha, tCrC, beta, tCgC);
}

// Host launcher
// MC=false (default): K-contiguous As (LayoutRight) + read-side swizzle
// MC=true:            M-contiguous As (LayoutLeft)  + write-side swizzle
template <int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 8, bool MC = false>
void sgemm_tiling(
    int m, int n, int k,
    float alpha,
    const float* A, int ldA,
    const float* B, int ldB,
    float beta,
    float* C, int ldC)
{
    using namespace cute;

    static_assert(BM % TM == 0 && BN % TN == 0, "BM/BN must be divisible by TM/TN");

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto shape_MNK = make_shape(m, n, k);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    constexpr int Tm = BM / TM;
    constexpr int Tn = BN / TN;

    // --- Swizzled smem layouts via tile_to_shape ---
    constexpr int atom_M = (Tm >= BK) ? Tm : BK;
    constexpr int SWZ_B = __builtin_ctz(BK);
    constexpr int SWZ_S = __builtin_ctz(atom_M);

    // MC=true: LayoutLeft (M contiguous) + write-side swizzle
    // MC=false: LayoutRight (K contiguous) + read-side swizzle
    auto sA_layout = [&]() {
        if constexpr (MC)
            return tile_to_shape(
                composition(Swizzle<SWZ_B, 0, SWZ_S>{},
                    make_layout(make_shape(Int<atom_M>{}, Int<BK>{}))),
                make_shape(Int<BM>{}, Int<BK>{}));
        else
            return tile_to_shape(
                composition(Swizzle<SWZ_B, 0, SWZ_S>{},
                    make_layout(make_shape(Int<atom_M>{}, Int<BK>{}), LayoutRight{})),
                make_shape(Int<BM>{}, Int<BK>{}));
    }();

    auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BK>{}), LayoutRight{});
    auto sC_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}));

    // --- gmem→smem tiled_copy ---
    using G2SCopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto g2s_copy_A = make_tiled_copy(G2SCopyAtom{},
        make_layout(make_shape(Int<Tm>{}, Int<Tn>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto g2s_copy_B = make_tiled_copy(G2SCopyAtom{},
        make_layout(make_shape(Int<Tm>{}, Int<Tn>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<1>{})));

    // --- tiled_mma: scalar FMA tiled across (Tm, Tn) threads ---
    auto mma = make_tiled_mma(UniversalFMA<float, float, float, float>{},
                              Layout<Shape<Int<Tm>, Int<Tn>, _1>>{});

    // --- smem→register copies aligned with MMA ---
    using S2RCopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto s2r_copy_A = make_tiled_copy_A(S2RCopyAtom{}, mma);
    auto s2r_copy_B = make_tiled_copy_B(S2RCopyAtom{}, mma);

    dim3 block_size(size(mma));
    dim3 grid_size(size(ceil_div(m, Int<BM>{})),
                   size(ceil_div(n, Int<BN>{})));

    sgemm_tiling_device<<<grid_size, block_size>>>(
        shape_MNK, cta_tiler, alpha,
        A, dA, g2s_copy_A, s2r_copy_A, sA_layout,
        B, dB, g2s_copy_B, s2r_copy_B, sB_layout,
        beta, C, dC, mma, sC_layout);
}
