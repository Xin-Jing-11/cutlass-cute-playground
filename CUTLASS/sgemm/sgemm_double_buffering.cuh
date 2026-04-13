#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm80.hpp>

/*
 * Double-buffered SGEMM using CuTe: D = alpha * A^T * B + beta * C   (TN layout)
 * A(M,K):(K,1), B(K,N):(1,K), C(M,N):(1,M).
 *
 * Uses cp.async for gmem→smem with double-buffered smem to overlap
 * data movement with computation.
 *
 * Pipeline schedule:
 *   prologue : issue tile 0 → fence → wait → sync
 *   mainloop : issue tile (i+1) → fence → compute tile i → wait → sync
 *   epilogue : compute final tile, write C
 */

template <class ProblemShape, class CtaTiler,
    class AStride, class TiledCopyA, class S2RCopyA, class ASmemLayout,
    class BStride, class TiledCopyB, class S2RCopyB, class BSmemLayout,
    class CStride, class TiledMMA, class CSmemLayout>
__global__ static
__launch_bounds__(decltype(size(TiledMMA{}))::value)
void sgemm_double_buffering_device(
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

    // Double-buffered smem
    constexpr int smemA_elem = cosize_v<decltype(sA_layout.layout_b())>;
    constexpr int smemB_elem = cosize_v<BSmemLayout>;
    __shared__ __align__(128) float smemA[2 * smemA_elem];
    __shared__ __align__(128) float smemB[2 * smemB_elem];

    // --- gmem→smem partitions ---
    auto thr_g2s_a = g2s_A.get_slice(threadIdx.x);
    Tensor tAgA = thr_g2s_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)

    auto thr_g2s_b = g2s_B.get_slice(threadIdx.x);
    Tensor tBgB = thr_g2s_b.partition_S(gB);

    // --- MMA partitions (use buffer 0 as reference for fragment shapes) ---
    Tensor sA_ref = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB_ref = make_tensor(make_smem_ptr(smemB), sB_layout);

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                     // (MMA, MMA_M, MMA_N)
    Tensor tCrA = thr_mma.partition_fragment_A(sA_ref);        // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB_ref);        // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);               // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // --- smem→register retile ---
    auto s2r_thr_a = s2r_A.get_slice(threadIdx.x);
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);                    // (CPY, MMA_M, MMA_K)

    auto s2r_thr_b = s2r_B.get_slice(threadIdx.x);
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);                    // (CPY, MMA_N, MMA_K)

    auto K_TILE_MAX  = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    int buf_read = 0, buf_write = 1;

    // ========== Prologue: load tile 0 into buffer 0 ==========
    {
        Tensor sA_w = make_tensor(make_smem_ptr(smemA), sA_layout);
        Tensor sB_w = make_tensor(make_smem_ptr(smemB), sB_layout);
        copy(g2s_A, tAgA(_, _, _, 0), thr_g2s_a.partition_D(sA_w));
        copy(g2s_B, tBgB(_, _, _, 0), thr_g2s_b.partition_D(sB_w));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
    }

    // ========== Mainloop: all but final tile ==========
    CUTE_NO_UNROLL
    for (int k_tile = 1; k_tile < K_TILE_MAX; ++k_tile) {
        // Issue async copies for next tile into write buffer
        {
            Tensor sA_w = make_tensor(make_smem_ptr(smemA + buf_write * smemA_elem), sA_layout);
            Tensor sB_w = make_tensor(make_smem_ptr(smemB + buf_write * smemB_elem), sB_layout);
            copy(g2s_A, tAgA(_, _, _, k_tile), thr_g2s_a.partition_D(sA_w));
            copy(g2s_B, tBgB(_, _, _, k_tile), thr_g2s_b.partition_D(sB_w));
            cp_async_fence();
        }

        // Compute on read buffer (overlaps with async copies)
        {
            Tensor sA_r = make_tensor(make_smem_ptr(smemA + buf_read * smemA_elem), sA_layout);
            Tensor sB_r = make_tensor(make_smem_ptr(smemB + buf_read * smemB_elem), sB_layout);
            Tensor tXsA = s2r_thr_a.partition_S(sA_r);
            Tensor tXsB = s2r_thr_b.partition_S(sB_r);
            CUTE_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                copy(s2r_A, tXsA(_, _, k_block), tXrA(_, _, k_block));
                copy(s2r_B, tXsB(_, _, k_block), tXrB(_, _, k_block));
                gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
            }
        }

        // Wait for async copies to complete
        cp_async_wait<0>();
        __syncthreads();

        buf_read ^= 1;
        buf_write ^= 1;
    }

    // ========== Epilogue: compute on final tile ==========
    {
        Tensor sA_r = make_tensor(make_smem_ptr(smemA + buf_read * smemA_elem), sA_layout);
        Tensor sB_r = make_tensor(make_smem_ptr(smemB + buf_read * smemB_elem), sB_layout);
        Tensor tXsA = s2r_thr_a.partition_S(sA_r);
        Tensor tXsB = s2r_thr_b.partition_S(sB_r);
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            copy(s2r_A, tXsA(_, _, k_block), tXrA(_, _, k_block));
            copy(s2r_B, tXsB(_, _, k_block), tXrB(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    axpby(alpha, tCrC, beta, tCgC);
}

// Host launcher
template <int BM = 128, int BN = 128, int BK = 16,
    int WM = 64, int WN = 64,
    int WMITER = 1, int WNITER = 4,
    int TM = 8, int TN = 4>
void sgemm_double_buffering(
    int m, int n, int k,
    float alpha,
    const float* A, int ldA,
    const float* B, int ldB,
    float beta,
    float* C, int ldC)
{
    using namespace cute;

    static_assert(BM % WM == 0 && BN % WN == 0, "BM/BN must be divisible by WM/WN");
    static_assert(WM % (WMITER * TM) == 0 && WN % (WNITER * TN) == 0);
    static_assert(BK % 4 == 0, "BK must be divisible by 4 for 128-bit vectorization");

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto shape_MNK = make_shape(m, n, k);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    constexpr int NWM = BM / WM;
    constexpr int NWN = BN / WN;
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int NTM = WSUBM / TM;
    constexpr int NTN = WSUBN / TN;
    static_assert(NTM * NTN == 32, "warp size must be 32");
    constexpr int NUM_THREADS = NWM * NWN * 32;
    constexpr int VEC = 4;
    constexpr int BK_VEC = BK / VEC;

    // --- sA swizzled with M=2 to preserve 128-bit store alignment ---
    constexpr int THR_M = NTM * NWM;
    constexpr int atom_M = (THR_M >= BK) ? THR_M : BK;
    constexpr int SWZ_M  = 2;
    constexpr int SWZ_B  = __builtin_ctz(BK) - SWZ_M;
    constexpr int SWZ_S  = __builtin_ctz(atom_M);
    static_assert(SWZ_B > 0, "BK must be > VEC=4 for M=2 swizzle");
    auto swizzle_atom = composition(Swizzle<SWZ_B, SWZ_M, SWZ_S>{},
        make_layout(make_shape(Int<atom_M>{}, Int<BK>{}), LayoutRight{}));
    auto sA_layout = tile_to_shape(swizzle_atom, make_shape(Int<BM>{}, Int<BK>{}));
    auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BK>{}), LayoutRight{});
    auto sC_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}));

    // --- gmem→smem: async 128-bit copies via cp.async ---
    constexpr int ThrK_A = BK_VEC;
    constexpr int ThrM_A = NUM_THREADS / ThrK_A;
    using G2SCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, float>;
    auto g2s_copy_A = make_tiled_copy(G2SCopyAtomA{},
        make_layout(make_shape(Int<ThrM_A>{}, Int<ThrK_A>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<VEC>{})));

    constexpr int ThrK_B = BK_VEC;
    constexpr int ThrN_B = NUM_THREADS / ThrK_B;
    using G2SCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, float>;
    auto g2s_copy_B = make_tiled_copy(G2SCopyAtomB{},
        make_layout(make_shape(Int<ThrN_B>{}, Int<ThrK_B>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<VEC>{})));

    // --- tiled_mma: warp-structured thread layout ---
    auto mma = make_tiled_mma(UniversalFMA<float, float, float, float>{},
        make_layout(
            make_shape(make_shape(Int<NTM>{}, Int<NWM>{}), make_shape(Int<NTN>{}, Int<NWN>{})),
            make_stride(make_stride(Int<1>{}, Int<NTM * NTN>{}), make_stride(Int<NTM>{}, Int<NWM * NTM * NTN>{}))
        ));

    // --- smem→register: scalar ---
    using S2RCopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto s2r_copy_A = make_tiled_copy_A(S2RCopyAtom{}, mma);
    auto s2r_copy_B = make_tiled_copy_B(S2RCopyAtom{}, mma);

    dim3 block_size(size(mma));
    dim3 grid_size(size(ceil_div(m, Int<BM>{})),
                   size(ceil_div(n, Int<BN>{})));

    sgemm_double_buffering_device<<<grid_size, block_size>>>(
        shape_MNK, cta_tiler, alpha,
        A, dA, g2s_copy_A, s2r_copy_A, sA_layout,
        B, dB, g2s_copy_B, s2r_copy_B, sB_layout,
        beta, C, dC, mma, sC_layout);
}
