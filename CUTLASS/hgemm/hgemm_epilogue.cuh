#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>

#include "scheduler.cuh"

/*
 * PERSISTENT CLUSTER-BASED HGEMM with TMA STORE EPILOGUE (CuTe/CUTLASS)
 * C = alpha * A^T * B + beta * C   (TN col-major).
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * Extends the persistent cluster-based design (hgemm_cluster) with a
 * TMA bulk store epilogue.  Consumers stage fp32 accumulators → fp16
 * into shared-memory sC, then one thread issues a single TMA bulk
 * store for the entire BM×BN output tile.
 *
 * Epilogue flow per tile:
 *   1. Each consumer warpgroup writes its accumulator rows to sC (smem).
 *   2. Consumer-only named-barrier sync — all rows committed.
 *   3. Thread 0 issues: fence → TMA store → commit → wait_read.
 *   4. Consumer-only named-barrier sync — smem safe to reuse next tile.
 *
 * Warp specialization:
 *   - WG0 (producer): TMA bulk loads into a QSIZE-deep pipeline.
 *   - WG1..NCS (consumers): WGMMA SS compute, each owns WM=64 rows.
 *
 * Cluster layout (CLUSTER_M × CLUSTER_N):
 *   - CTAs along M share B tile → B multicast when CLUSTER_M > 1.
 *   - CTAs along N share A tile → A multicast when CLUSTER_N > 1.
 */

// ---------------------------------------------------------------------------
// Shared storage
// ---------------------------------------------------------------------------

template <int BM, int BN, int BK, int QSIZE>
struct HgemmEpilogueSharedStorage {
    alignas(128) cute::half_t smemA[QSIZE * BM * BK];
    alignas(128) cute::half_t smemB[QSIZE * BN * BK];
    alignas(128) cute::half_t smemC[BM * BN];       // epilogue staging buffer
    alignas(8)   uint64_t     full_barrier[QSIZE];   // producer → consumer (TX)
    alignas(8)   uint64_t     empty_barrier[QSIZE];  // consumer → producer (cross-CTA)
};

// ---------------------------------------------------------------------------
// Device kernel — K-loop copied from hgemm_cluster, epilogue added
// ---------------------------------------------------------------------------

template <int BM, int BN, int BK, int NCS, int QSIZE, int NSM,
          int CLUSTER_M, int CLUSTER_N,
          class TmaA, class ASmemLayout,
          class TmaB, class BSmemLayout,
          class ASubLayout, class BSubLayout,
          class TmaC,
          class CStride, class TiledMMA>
__global__ static
__launch_bounds__((NCS + 1) * 128)
__cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
void hgemm_epilogue_device(
    int M, int N, int K,
    float alpha, float beta,
    CUTLASS_GRID_CONSTANT TmaA const tma_a, ASmemLayout sA_layout,
    CUTLASS_GRID_CONSTANT TmaB const tma_b, BSmemLayout sB_layout,
    ASubLayout sA_sub_layout, BSubLayout sB_sub_layout,
    CUTLASS_GRID_CONSTANT TmaC const tma_store_c,
    cute::half_t const* C_gmem_ptr, CStride dC,
    TiledMMA mma)
{
    using namespace cute;

    constexpr int WM       = 64;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;
    static_assert(BM == WM * NCS);

    // ------------------------------------------------------------------
    // Shared storage
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) char smem_raw[];
    auto& shared = *reinterpret_cast<
        HgemmEpilogueSharedStorage<BM, BN, BK, QSIZE>*>(smem_raw);

    // Rank-3 smem tensors for TMA partitioning: (tile, BK, QSIZE)
    Tensor sA = make_tensor(make_smem_ptr(shared.smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(shared.smemB), sB_layout);

    // ------------------------------------------------------------------
    // Cluster identity
    // ------------------------------------------------------------------
    uint32_t cluster_rank = cute::block_rank_in_cluster();
    uint32_t cluster_id   = blockIdx.x / NCLUSTER;
    uint32_t rank_m       = cluster_rank / CLUSTER_N;
    uint32_t rank_n       = cluster_rank % CLUSTER_N;

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
            EmptyBarrier::init(&shared.empty_barrier[s], NCS * NCLUSTER + 1);
        }
        cutlass::arch::fence_barrier_init();
    }
    cute::cluster_sync();

    // ------------------------------------------------------------------
    // Scheduler: cluster-level super-blocks
    // ------------------------------------------------------------------
    constexpr int SUPER_BM = BM * CLUSTER_M;
    constexpr int SUPER_BN = BN * CLUSTER_N;
    scheduler::Schedule<1, NSM / NCLUSTER, SUPER_BM, SUPER_BN,
                        16 / CLUSTER_M, 8 / CLUSTER_N>
        schedule(M, N, cluster_id);

    // ------------------------------------------------------------------
    // TMA transaction bytes
    // ------------------------------------------------------------------
    constexpr int tma_tx_bytes = BM * BK * (int)sizeof(cute::half_t)
                               + BN * BK * (int)sizeof(cute::half_t);

    // ------------------------------------------------------------------
    // Multicast masks
    // ------------------------------------------------------------------
    uint16_t a_mcast_mask = 0;
    if constexpr (CLUSTER_N > 1) {
        a_mcast_mask = uint16_t((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
    }
    uint16_t b_mcast_mask = 0;
    if constexpr (CLUSTER_M > 1) {
        for (int i = 0; i < CLUSTER_M; ++i)
            b_mcast_mask |= uint16_t(1) << (i * CLUSTER_N + rank_n);
    }

    // ==================================================================
    // PRODUCER (WG0) — identical to hgemm_cluster
    // ==================================================================
    if (warp_group_idx == 0) {
        constexpr int NRG_PROD = (NCS <= 2 ? 24 : 32);
        cutlass::arch::warpgroup_reg_dealloc<NRG_PROD>();

        if (tid_in_wg == 0) {
            int gq = 0, phase = 0;
            int tile_id;
            while ((tile_id = schedule.next()) != -1) {
                int super_n = tile_id % (N / SUPER_BN);
                int super_m = tile_id / (N / SUPER_BN);
                int bm = (super_m * CLUSTER_M + rank_m) * BM;
                int bn = (super_n * CLUSTER_N + rank_n) * BN;

                Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
                Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));

                Tensor gA = local_tile(mA, make_shape(Int<BM>{}, Int<BK>{}),
                                       make_coord(bm / BM, _));
                Tensor gB = local_tile(mB, make_shape(Int<BN>{}, Int<BK>{}),
                                       make_coord(bn / BN, _));

                auto [tAgA, tAsA] = tma_partition(
                    tma_a, Int<0>{}, Layout<_1>{},
                    group_modes<0, 2>(sA), group_modes<0, 2>(gA));
                auto [tBgB, tBsB] = tma_partition(
                    tma_b, Int<0>{}, Layout<_1>{},
                    group_modes<0, 2>(sB), group_modes<0, 2>(gB));

                int k_tile_count = size<1>(tAgA);

                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int qid = gq % QSIZE;

                    EmptyBarrier::arrive(&shared.empty_barrier[qid]);
                    EmptyBarrier::wait(&shared.empty_barrier[qid], phase);

                    FullBarrier::arrive_and_expect_tx(
                        &shared.full_barrier[qid], tma_tx_bytes);

                    if constexpr (CLUSTER_N > 1) {
                        if (rank_n == 0) {
                            copy(tma_a.with(shared.full_barrier[qid], a_mcast_mask),
                                 tAgA(_, k_tile), tAsA(_, qid));
                        }
                    } else {
                        copy(tma_a.with(shared.full_barrier[qid]),
                             tAgA(_, k_tile), tAsA(_, qid));
                    }

                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0) {
                            copy(tma_b.with(shared.full_barrier[qid], b_mcast_mask),
                                 tBgB(_, k_tile), tBsB(_, qid));
                        }
                    } else {
                        copy(tma_b.with(shared.full_barrier[qid]),
                             tBgB(_, k_tile), tBsB(_, qid));
                    }

                    gq++;
                    if (gq % QSIZE == 0) phase ^= 1;
                }
            }
        }

        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
        return;
    }

    // ==================================================================
    // CONSUMER (WG1..NCS) — K-loop from hgemm_cluster, TMA store epilogue
    // ==================================================================
    constexpr int NRG_CONS = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
    cutlass::arch::warpgroup_reg_alloc<NRG_CONS>();

    int consumer_thread = tid_in_wg;
    int csid            = warp_group_idx - 1;

    // Initial empty signal
    if (consumer_thread == 0) {
        for (int s = 0; s < QSIZE; ++s) {
            for (uint32_t c = 0; c < NCLUSTER; ++c) {
                if (c == cluster_rank) {
                    EmptyBarrier::arrive(&shared.empty_barrier[s]);
                } else {
                    EmptyBarrier::arrive(&shared.empty_barrier[s], c, 1);
                }
            }
        }
    }

    // MMA setup
    auto thr_mma = mma.get_slice(consumer_thread);

    // Accumulator fragment
    Tensor mC_ref = make_tensor(make_gmem_ptr(C_gmem_ptr), make_shape(M, N), dC);
    Tensor gC_ref = local_tile(mC_ref, make_shape(Int<WM>{}, Int<BN>{}),
                               make_coord(0, 0));
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_ref));

    // Epilogue named barrier (all consumer threads)
    cutlass::arch::NamedBarrier epi_barrier(NCS * 128,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    // Per-warpgroup sC sub-view layout (WM rows, BM stride for col-major)
    auto sC_sub_layout = make_layout(make_shape(Int<WM>{}, Int<BN>{}),
                                     make_stride(Int<1>{}, Int<BM>{}));

    int tile_id;
    int gq_cons = 0, phase_cons = 0;
    while ((tile_id = schedule.next()) != -1) {
        int super_n = tile_id % (N / SUPER_BN);
        int super_m = tile_id / (N / SUPER_BN);
        int bm = (super_m * CLUSTER_M + rank_m) * BM;
        int bn = (super_n * CLUSTER_N + rank_n) * BN;

        // Global C for this consumer's WM rows (for reading beta*C)
        Tensor mC = make_tensor(make_gmem_ptr(C_gmem_ptr), make_shape(M, N), dC);
        Tensor gC = local_tile(mC, make_shape(Int<WM>{}, Int<BN>{}),
                               make_coord(bm / WM + csid, bn / BN));
        Tensor tCgC = thr_mma.partition_C(gC);

        clear(tCrC);

        int k_tile_count = K / BK;

        // --- K-loop: WGMMA accumulate (identical to hgemm_cluster) ---
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            int qid = gq_cons % QSIZE;

            FullBarrier::wait(&shared.full_barrier[qid], phase_cons);

            Tensor sA_cs = make_tensor(
                make_smem_ptr(shared.smemA + qid * BM * BK + csid * WM * BK),
                sA_sub_layout);
            Tensor sB_stg = make_tensor(
                make_smem_ptr(shared.smemB + qid * BN * BK),
                sB_sub_layout);

            auto tCsA = thr_mma.partition_A(sA_cs);
            auto tCsB = thr_mma.partition_B(sB_stg);

            warpgroup_fence_operand(tCrC);
            warpgroup_arrive();
            gemm(mma, tCsA, tCsB, tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrC);

            if (consumer_thread == 0) {
                for (uint32_t c = 0; c < NCLUSTER; ++c) {
                    if (c == cluster_rank) {
                        EmptyBarrier::arrive(&shared.empty_barrier[qid]);
                    } else {
                        EmptyBarrier::arrive(&shared.empty_barrier[qid], c, 1);
                    }
                }
            }

            gq_cons++;
            if (gq_cons % QSIZE == 0) phase_cons ^= 1;
        }

        // === TMA STORE EPILOGUE ===

        // Step 1: Stage accumulators (fp32 → fp16) into sC (col-major)
        {
            Tensor sC_wg = make_tensor(
                make_smem_ptr(shared.smemC + csid * WM), sC_sub_layout);
            Tensor tCsC = thr_mma.partition_C(sC_wg);

            // D = alpha * acc + beta * C
            CUTE_UNROLL
            for (int i = 0; i < size(tCrC); ++i) {
                tCsC(i) = cute::half_t(alpha * tCrC(i) + beta * float(tCgC(i)));
            }
        }

        // Step 2: Sync all consumers — sC fully written
        epi_barrier.sync();

        // Step 3: One thread issues TMA bulk store: sC → global C
        if (csid == 0 && consumer_thread == 0) {
            cutlass::arch::fence_view_async_shared();

            Tensor sC_full = make_tensor(make_smem_ptr(shared.smemC),
                make_layout(make_shape(Int<BM>{}, Int<BN>{}),
                            make_stride(Int<1>{}, Int<BM>{})));
            Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
            Tensor gC_store = local_tile(mC_tma, make_shape(Int<BM>{}, Int<BN>{}),
                                         make_coord(bm / BM, bn / BN));
            auto [tSgC, tSsC] = tma_partition(tma_store_c, Int<0>{}, Layout<_1>{},
                                               group_modes<0,2>(sC_full),
                                               group_modes<0,2>(gC_store));
            copy(tma_store_c, tSsC, tSgC);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }

        // Step 4: Sync — smem sC safe to reuse
        epi_barrier.sync();
    }

    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}


// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

template <int BM = 128, int BN = 256, int BK = 64, int NCS = 2, int QSIZE = 2,
          int CLUSTER_M = 2, int CLUSTER_N = 1>
void hgemm_epilogue(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    static_assert(NCS >= 1);
    constexpr int WM       = 64;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;
    constexpr int NSM      = 128;
    static_assert(BM == WM * NCS, "BM must equal 64 * NCS");
    static_assert(NSM % NCLUSTER == 0);

    // TN strides
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // WGMMA atom
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
            static_assert(BN == 64 || BN == 128 || BN == 256);
        }
    }();

    // Smem layouts: rank-3 for TMA (tile, BK, QSIZE)
    using SmemAtom = GMMA::Layout_K_SW128_Atom<cute::half_t>;
    auto sA_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<QSIZE>{}));
    auto sB_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<QSIZE>{}));
    auto sA_sub_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<WM>{}, Int<BK>{}));
    auto sB_sub_layout = tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}));

    // Col-major sC layout for TMA store
    auto sC_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}),
                                 make_stride(Int<1>{}, Int<BM>{}));

    // Global tensors for TMA descriptors
    Tensor mA_desc = make_tensor(A, make_shape(m, k), dA);
    Tensor mB_desc = make_tensor(B, make_shape(n, k), dB);
    Tensor mC_desc = make_tensor(C, make_shape(m, n), dC);

    // TMA load descriptors — rank-2 smem slice for descriptor creation
    auto tmaA = [&]() {
        if constexpr (CLUSTER_N > 1) {
            return make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mA_desc,
                                 sA_layout(_, _, 0),
                                 make_shape(Int<BM>{}, Int<BK>{}), Int<1>{});
        } else {
            return make_tma_atom(SM90_TMA_LOAD{}, mA_desc,
                                 sA_layout(_, _, 0),
                                 make_shape(Int<BM>{}, Int<BK>{}));
        }
    }();

    auto tmaB = [&]() {
        if constexpr (CLUSTER_M > 1) {
            return make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mB_desc,
                                 sB_layout(_, _, 0),
                                 make_shape(Int<BN>{}, Int<BK>{}), Int<1>{});
        } else {
            return make_tma_atom(SM90_TMA_LOAD{}, mB_desc,
                                 sB_layout(_, _, 0),
                                 make_shape(Int<BN>{}, Int<BK>{}));
        }
    }();

    // TMA store descriptor for C
    auto tmaC = make_tma_atom(SM90_TMA_STORE{}, mC_desc, sC_layout,
                              make_shape(Int<BM>{}, Int<BN>{}));

    // Launch
    constexpr int NUM_THREADS = (NCS + 1) * 128;
    constexpr int smem_bytes  = sizeof(
        HgemmEpilogueSharedStorage<BM, BN, BK, QSIZE>);

    auto kernel = hgemm_epilogue_device<BM, BN, BK, NCS, QSIZE, NSM,
        CLUSTER_M, CLUSTER_N,
        decltype(tmaA), decltype(sA_layout),
        decltype(tmaB), decltype(sB_layout),
        decltype(sA_sub_layout), decltype(sB_sub_layout),
        decltype(tmaC),
        decltype(dC), decltype(mma)>;

    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dim3 block(NUM_THREADS);
    dim3 grid(NSM);

    kernel<<<grid, block, smem_bytes>>>(
        m, n, k, alpha, beta,
        tmaA, sA_layout,
        tmaB, sB_layout,
        sA_sub_layout, sB_sub_layout,
        tmaC,
        static_cast<cute::half_t const*>(C), dC, mma);
}
