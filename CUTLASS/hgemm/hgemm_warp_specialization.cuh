#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>    // GMMA::Layout_K_SW128_Atom, SM90_64x*
#include <cute/arch/copy_sm90_tma.hpp>            // SM90_TMA_LOAD
#include <cute/atom/copy_traits_sm90_tma.hpp>     // make_tma_atom, tma_partition
#include <cutlass/arch/barrier.h>                 // ClusterBarrier, ClusterTransactionBarrier
#include <cutlass/device_kernel.h>                // CUTLASS_GRID_CONSTANT

/*
 * Warp-specialized WGMMA+TMA HGEMM: C = alpha * A^T * B + beta * C  (TN layout)
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * (NCS+1) warp groups of 128 threads each:
 *   - WG0:        TMA producer.  elect_one thread issues cp.async.bulk.tensor
 *                 for A and B into a QSIZE-deep circular queue in smem.
 *   - WG1..NCS:   WGMMA consumers.  Each consumer owns WM=64 rows of A/C and
 *                 the full BN columns.  WGMMA SS (shared-shared) MMA atoms.
 *
 * Synchronization:
 *   - full_barrier[s]  (ClusterTransactionBarrier): producer → consumer
 *   - empty_barrier[s] (ClusterBarrier):            consumer → producer
 *
 * BM must equal 64 * NCS.
 */

template <int BM, int BN, int BK, int QSIZE>
struct HgemmWarpSpecSharedStorage {
    alignas(128) cute::half_t smemA[QSIZE * BM * BK];
    alignas(128) cute::half_t smemB[QSIZE * BN * BK];
    alignas(8)   uint64_t     full_barrier[QSIZE];
    alignas(8)   uint64_t     empty_barrier[QSIZE];
};

template <int BM, int BN, int BK, int NCS, int QSIZE,
          class TmaA, class ASmemLayout,
          class TmaB, class BSmemLayout,
          class ASubLayout, class BSubLayout,
          class CStride, class TiledMMA>
__global__ static
__launch_bounds__((NCS + 1) * 128)
void hgemm_warp_spec_device(
    int M, int N, int K,
    float alpha,
    CUTLASS_GRID_CONSTANT TmaA const tma_a, ASmemLayout sA_layout,
    CUTLASS_GRID_CONSTANT TmaB const tma_b, BSmemLayout sB_layout,
    ASubLayout sA_sub_layout, BSubLayout sB_sub_layout,
    float beta,
    cute::half_t* C, CStride dC,
    TiledMMA mma)
{
    using namespace cute;

    constexpr int WM = 64;
    static_assert(BM == WM * NCS);

    // ------------------------------------------------------------------
    // Shared storage
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) char smem_raw[];
    auto& shared = *reinterpret_cast<
        HgemmWarpSpecSharedStorage<BM, BN, BK, QSIZE>*>(smem_raw);

    // Rank-3 smem tensors for TMA partitioning: (tile_M/N, BK, QSIZE)
    Tensor sA = make_tensor(make_smem_ptr(shared.smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(shared.smemB), sB_layout);

    // ------------------------------------------------------------------
    // Global tensors from TMA descriptors + tiling
    // ------------------------------------------------------------------
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));   // (M, K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));   // (N, K)

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BM, BK, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BN, BK, k)

    // ------------------------------------------------------------------
    // TMA partition (group (tile, BK) into a single TMA mode)
    // ------------------------------------------------------------------
    auto [tAgA, tAsA] = tma_partition(
        tma_a, Int<0>{}, Layout<_1>{},
        group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(
        tma_b, Int<0>{}, Layout<_1>{},
        group_modes<0, 2>(sB), group_modes<0, 2>(gB));

    constexpr int tma_tx_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                               + sizeof(make_tensor_like(tensor<0>(tBsB)));

    int k_tile_count = size<1>(tAgA);

    // ------------------------------------------------------------------
    // Role identification
    // ------------------------------------------------------------------
    int warp_group_idx = threadIdx.x / 128;
    int tid_in_wg      = threadIdx.x % 128;

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    // ------------------------------------------------------------------
    // Barrier init
    // ------------------------------------------------------------------
    if (threadIdx.x == 0) {
        CUTE_UNROLL
        for (int s = 0; s < QSIZE; ++s) {
            FullBarrier::init(&shared.full_barrier[s], 1);
            EmptyBarrier::init(&shared.empty_barrier[s], NCS);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Pre-arrive NCS times on every empty_barrier to mark all stages empty
    if (threadIdx.x == 0) {
        CUTE_UNROLL
        for (int s = 0; s < QSIZE; ++s) {
            CUTE_UNROLL
            for (int i = 0; i < NCS; ++i) {
                EmptyBarrier::arrive(&shared.empty_barrier[s]);
            }
        }
    }

    // ==================================================================
    // PRODUCER (WG0, 128 threads — only thread 0 issues TMAs)
    // ==================================================================
    if (warp_group_idx == 0) {
        constexpr int NRG_PROD = (NCS <= 2 ? 24 : 32);
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(NRG_PROD));

        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            int stage = k_tile % QSIZE;
            int phase = (k_tile / QSIZE) % 2;

            EmptyBarrier::wait(&shared.empty_barrier[stage], phase);

            if (threadIdx.x == 0) {
                FullBarrier::arrive_and_expect_tx(
                    &shared.full_barrier[stage], tma_tx_bytes);
                copy(tma_a.with(shared.full_barrier[stage]),
                     tAgA(_, k_tile), tAsA(_, stage));
                copy(tma_b.with(shared.full_barrier[stage]),
                     tBgB(_, k_tile), tBsB(_, stage));
            }
        }
        return;
    }

    // ==================================================================
    // CONSUMER (WG1 .. WG_NCS)
    // ==================================================================
    constexpr int NRG_CONS = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(NRG_CONS));

    int consumer_thread = tid_in_wg;          // 0..127
    int csid            = warp_group_idx - 1; // 0..NCS-1

    // gC for this consumer: WM rows starting at blockIdx.x * BM + csid * WM
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    Tensor gC = local_tile(mC, make_shape(Int<WM>{}, Int<BN>{}),
                           make_coord(blockIdx.x * NCS + csid, blockIdx.y));

    auto thr_mma = mma.get_slice(consumer_thread);
    Tensor tCgC = thr_mma.partition_C(gC);            // (MMA, MMA_M, MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);      // FP32 accumulator
    clear(tCrC);

    // ------------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------------
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int stage = k_tile % QSIZE;
        int phase = (k_tile / QSIZE) % 2;

        FullBarrier::wait(&shared.full_barrier[stage], phase);

        // Per-consumer, per-stage 2-D smem sub-views
        Tensor sA_cs = make_tensor(
            make_smem_ptr(shared.smemA + stage * BM * BK + csid * WM * BK),
            sA_sub_layout);                                   // (WM, BK)
        Tensor sB_stg = make_tensor(
            make_smem_ptr(shared.smemB + stage * BN * BK),
            sB_sub_layout);                                   // (BN, BK)

        auto tCsA = thr_mma.partition_A(sA_cs);
        auto tCsB = thr_mma.partition_B(sB_stg);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(mma, tCsA, tCsB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        // Signal producer that this stage is free
        if (consumer_thread == 0) {
            EmptyBarrier::arrive(&shared.empty_barrier[stage]);
        }
    }

    // ------------------------------------------------------------------
    // Epilogue
    // ------------------------------------------------------------------
    axpby(alpha, tCrC, beta, tCgC);
}


// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 256, int BK = 64, int NCS = 2, int QSIZE = 4>
void hgemm_warp_spec(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    constexpr int WM = 64;
    static_assert(BM == WM * NCS, "BM must equal 64 * NCS");

    // TN strides
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // ---------------------------------------------------------------
    // WGMMA atom selection (SS: both operands in smem, K-major)
    // ---------------------------------------------------------------
    auto mma = []() {
        if constexpr (BN == 64) {
            return make_tiled_mma(
                SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else if constexpr (BN == 128) {
            return make_tiled_mma(
                SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else if constexpr (BN == 256) {
            return make_tiled_mma(
                SM90_64x256x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else {
            static_assert(BN == 64 || BN == 128 || BN == 256,
                          "Unsupported BN for WGMMA");
        }
    }();

    // ---------------------------------------------------------------
    // Smem layouts: GMMA K-major 128-byte swizzle
    // ---------------------------------------------------------------
    using SmemAtom = GMMA::Layout_K_SW128_Atom<cute::half_t>;

    // Rank-3 layouts for TMA partitioning (tile, BK, QSIZE)
    auto sA_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<QSIZE>{}));
    auto sB_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<QSIZE>{}));

    // Rank-2 sub-layouts for consumer WGMMA partitions
    auto sA_sub_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<WM>{}, Int<BK>{}));
    auto sB_sub_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}));

    // ---------------------------------------------------------------
    // TMA descriptors (built from a single-stage 2-D slice)
    // ---------------------------------------------------------------
    Tensor mA_desc = make_tensor(A, make_shape(m, k), dA);
    Tensor mB_desc = make_tensor(B, make_shape(n, k), dB);

    auto tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA_desc, sA_layout(_, _, 0),
                              make_shape(Int<BM>{}, Int<BK>{}));
    auto tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB_desc, sB_layout(_, _, 0),
                              make_shape(Int<BN>{}, Int<BK>{}));

    // ---------------------------------------------------------------
    // Launch config
    // ---------------------------------------------------------------
    constexpr int NUM_THREADS = (NCS + 1) * 128;
    constexpr int smem_bytes  = sizeof(HgemmWarpSpecSharedStorage<BM, BN, BK, QSIZE>);

    auto kernel = hgemm_warp_spec_device<BM, BN, BK, NCS, QSIZE,
        decltype(tmaA), decltype(sA_layout),
        decltype(tmaB), decltype(sB_layout),
        decltype(sA_sub_layout), decltype(sB_sub_layout),
        decltype(dC), decltype(mma)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    dim3 block(NUM_THREADS);
    dim3 grid(size(ceil_div(m, Int<BM>{})),
              size(ceil_div(n, Int<BN>{})));

    kernel<<<grid, block, smem_bytes>>>(
        m, n, k, alpha,
        tmaA, sA_layout,
        tmaB, sB_layout,
        sA_sub_layout, sB_sub_layout,
        beta, C, dC, mma);
}
