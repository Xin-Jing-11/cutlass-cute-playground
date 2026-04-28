#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>    // GMMA::Layout_K_SW128_Atom, SM90_64x*
#include <cute/arch/copy_sm90_tma.hpp>            // SM90_TMA_LOAD
#include <cute/atom/copy_traits_sm90_tma.hpp>     // make_tma_atom, tma_partition
#include <cutlass/arch/barrier.h>                 // ClusterBarrier, ClusterTransactionBarrier
#include <cutlass/device_kernel.h>                // CUTLASS_GRID_CONSTANT
#include "scheduler.cuh"

/*
 * PERSISTENT WARP-SPECIALIZED HGEMM with TMA + WGMMA (CuTe)
 * C = alpha * A^T * B + beta * C   (TN col-major).
 *
 * Persistent kernel: launches exactly NSM thread blocks, each looping
 * over multiple output tiles via a super-tiled scheduler (Schedule<1>)
 * that groups TM×TN tiles for L2 cache locality.
 *
 * Warp specialization:
 *   - WG0:        TMA producer.  Thread 0 issues cp.async.bulk.tensor
 *                 for A and B into a QSIZE-deep circular queue in smem.
 *   - WG1..NCS:   WGMMA consumers.  Each consumer owns WM=64 rows of A/C
 *                 and the full BN columns.  WGMMA SS (shared-shared) MMA atoms.
 *
 * Synchronization:
 *   - full_barrier[s]  (ClusterTransactionBarrier): producer → consumer
 *   - empty_barrier[s] (ClusterBarrier):            consumer → producer
 *
 * Store/load overlap: because the persistent loop reuses the same thread
 * block across tiles, the producer can begin loading the next tile's data
 * while consumers are still writing the previous tile's epilogue to global
 * memory—hiding store latency behind TMA load latency.
 *
 * BM must equal 64 * NCS.
 */

template <int BM, int BN, int BK, int QSIZE>
struct HgemmPersistentSharedStorage {
    alignas(128) cute::half_t smemA[QSIZE * BM * BK];
    alignas(128) cute::half_t smemB[QSIZE * BN * BK];
    alignas(8)   uint64_t     full_barrier[QSIZE];
    alignas(8)   uint64_t     empty_barrier[QSIZE];
};

template <int BM, int BN, int BK, int NCS, int QSIZE, int NSM,
          class TmaA, class ASmemLayout,
          class TmaB, class BSmemLayout,
          class ASubLayout, class BSubLayout,
          class CStride, class TiledMMA>
__global__ static
__launch_bounds__((NCS + 1) * 128)
void hgemm_persistent_device(
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
        HgemmPersistentSharedStorage<BM, BN, BK, QSIZE>*>(smem_raw);

    constexpr int tma_tx_bytes = (BM * BK + BN * BK) * int(sizeof(cute::half_t));

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

    scheduler::Schedule<1, NSM, BM, BN, 16, 8> schedule(M, N, blockIdx.x);

    // ==================================================================
    // PRODUCER (WG0 — only thread 0 issues TMAs)
    // ==================================================================
    if (warp_group_idx == 0) {
        constexpr int NRG_PROD = (NCS <= 2 ? 24 : 32);
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(NRG_PROD));

        if (threadIdx.x == 0) {
            int global_k = 0;
            int tile_id;

            while ((tile_id = schedule.next()) != -1) {
                int tile_n = tile_id % (N / BN);
                int tile_m = tile_id / (N / BN);
                int bm = tile_m * BM;
                int bn = tile_n * BN;

                for (int bk = 0; bk < K; bk += BK, ++global_k) {
                    int stage = global_k % QSIZE;
                    int phase = (global_k / QSIZE) % 2;

                    EmptyBarrier::wait(&shared.empty_barrier[stage], phase);

                    FullBarrier::arrive_and_expect_tx(
                        &shared.full_barrier[stage], tma_tx_bytes);
                    SM90_TMA_LOAD_2D::copy(
                        tma_a.get_tma_descriptor(),
                        &shared.full_barrier[stage],
                        uint64_t(0),
                        &shared.smemA[stage * BM * BK],
                        bk, bm);
                    SM90_TMA_LOAD_2D::copy(
                        tma_b.get_tma_descriptor(),
                        &shared.full_barrier[stage],
                        uint64_t(0),
                        &shared.smemB[stage * BN * BK],
                        bk, bn);
                }
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

    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto thr_mma = mma.get_slice(consumer_thread);

    // Create accumulator ONCE (same shape for every tile) — keeps
    // registers pinned so the compiler can pipeline WGMMA correctly.
    Tensor gC_ref = local_tile(mC, make_shape(Int<WM>{}, Int<BN>{}),
                               make_coord(csid, 0));
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_ref));

    int global_k = 0;
    int tile_id;
    int K_TILES = K / BK;

    while ((tile_id = schedule.next()) != -1) {
        int tile_n = tile_id % (N / BN);
        int tile_m = tile_id / (N / BN);

        clear(tCrC);
        warpgroup_fence_operand(tCrC);

        for (int kt = 0; kt < K_TILES; ++kt, ++global_k) {
            int stage = global_k % QSIZE;
            int phase = (global_k / QSIZE) % 2;

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

        // Epilogue: write this tile's result to global memory
        warpgroup_fence_operand(tCrC);
        Tensor gC = local_tile(mC, make_shape(Int<WM>{}, Int<BN>{}),
                               make_coord(tile_m * NCS + csid, tile_n));
        Tensor tCgC = thr_mma.partition_C(gC);
        axpby(alpha, tCrC, beta, tCgC);
        warpgroup_fence_operand(tCrC);
    }
}


// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 256, int BK = 64, int NCS = 2, int QSIZE = 4>
void hgemm_persistent(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    constexpr int WM  = 64;
    constexpr int NSM = 128;
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
    constexpr int smem_bytes  = sizeof(
        HgemmPersistentSharedStorage<BM, BN, BK, QSIZE>);

    auto kernel = hgemm_persistent_device<BM, BN, BK, NCS, QSIZE, NSM,
        decltype(tmaA), decltype(sA_layout),
        decltype(tmaB), decltype(sB_layout),
        decltype(sA_sub_layout), decltype(sB_sub_layout),
        decltype(dC), decltype(mma)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    dim3 block(NUM_THREADS);
    dim3 grid(NSM);

    kernel<<<grid, block, smem_bytes>>>(
        m, n, k, alpha,
        tmaA, sA_layout,
        tmaB, sB_layout,
        sA_sub_layout, sB_sub_layout,
        beta, C, dC, mma);
}
