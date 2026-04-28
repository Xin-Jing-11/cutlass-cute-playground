#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include "scheduler.cuh"

/*
 * PERSISTENT CLUSTER-BASED WGMMA+TMA HGEMM with TMA MULTICAST
 * C = alpha * A^T * B + beta * C   (TN col-major).
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * Extends the persistent warp-specialized design with thread block clusters.
 * CTAs within a cluster share a virtual address space via DSMEM, enabling
 * TMA multicast: a single TMA load broadcasts the same tile to all CTAs in
 * the cluster, cutting global memory traffic by the cluster size for the
 * shared operand.
 *
 * Cluster layout (CLUSTER_M × CLUSTER_N):
 *   - CTAs along M share the same B tile → B is multicast when CLUSTER_M > 1.
 *   - CTAs along N share the same A tile → A is multicast when CLUSTER_N > 1.
 *
 * Warp specialization:
 *   - WG0 (producer):     TMA bulk copies through a QSIZE-deep circular queue.
 *                          For the shared operand, only rank-0 along the multicast
 *                          dimension issues the multicast TMA.
 *   - WG1..NCS (consumers): WGMMA SS MMA, each owning WM=64 rows.
 *
 * Barrier design:
 *   - full[QSIZE]: local (ClusterTransactionBarrier). 1 producer arrive_and_expect_tx.
 *     TMA multicast delivers data + barrier signal to every destination CTA.
 *   - empty[QSIZE]: cross-CTA (ClusterBarrier). NCS*NCLUSTER arrivals (1 per consumer
 *     warpgroup per CTA across the cluster).
 */

template <int BM, int BN, int BK, int QSIZE>
struct HgemmClusterSharedStorage {
    alignas(128) cute::half_t smemA[QSIZE * BM * BK];
    alignas(128) cute::half_t smemB[QSIZE * BN * BK];
    alignas(8)   uint64_t     full_barrier[QSIZE];
    alignas(8)   uint64_t     empty_barrier[QSIZE];
};

template <int BM, int BN, int BK, int NCS, int QSIZE, int NSM,
          int CLUSTER_M, int CLUSTER_N,
          class TmaA, class ASmemLayout,
          class TmaB, class BSmemLayout,
          class ASubLayout, class BSubLayout,
          class CStride, class TiledMMA>
__global__ static
__launch_bounds__((NCS + 1) * 128)
__cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
void hgemm_cluster_device(
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

    constexpr int WM       = 64;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;
    static_assert(BM == WM * NCS);

    // ------------------------------------------------------------------
    // Shared storage
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) char smem_raw[];
    auto& shared = *reinterpret_cast<
        HgemmClusterSharedStorage<BM, BN, BK, QSIZE>*>(smem_raw);

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
    // All CTAs must finish barrier init before any CTA accesses remote barriers.
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
    // TMA transaction bytes (per-stage, A + B)
    // ------------------------------------------------------------------
    constexpr int tma_tx_bytes = BM * BK * (int)sizeof(cute::half_t)
                               + BN * BK * (int)sizeof(cute::half_t);

    // ------------------------------------------------------------------
    // Multicast masks (constant for entire kernel)
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
    // PRODUCER (WG0, 128 threads — only thread 0 issues TMAs)
    // ==================================================================
    if (warp_group_idx == 0) {
        constexpr int NRG_PROD = (NCS <= 2 ? 24 : 32);
        cutlass::arch::warpgroup_reg_dealloc<NRG_PROD>();

        if (tid_in_wg == 0) {
            int gq = 0, phase = 0;
            int tile_id;
            while ((tile_id = schedule.next()) != -1) {
                // Decode super-block tile → per-CTA element offsets
                int super_n = tile_id % (N / SUPER_BN);
                int super_m = tile_id / (N / SUPER_BN);
                int bm = (super_m * CLUSTER_M + rank_m) * BM;
                int bn = (super_n * CLUSTER_N + rank_n) * BN;

                // TMA global views (shape from descriptor)
                Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
                Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));

                // Tile for this CTA's m/n position
                Tensor gA = local_tile(mA, make_shape(Int<BM>{}, Int<BK>{}),
                                       make_coord(bm / BM, _));
                Tensor gB = local_tile(mB, make_shape(Int<BN>{}, Int<BK>{}),
                                       make_coord(bn / BN, _));

                // TMA partition — no multicast partitioning (cluster_size=1 in
                // the TMA descriptor), each CTA receives the full tile.
                auto [tAgA, tAsA] = tma_partition(
                    tma_a, Int<0>{}, Layout<_1>{},
                    group_modes<0, 2>(sA), group_modes<0, 2>(gA));
                auto [tBgB, tBsB] = tma_partition(
                    tma_b, Int<0>{}, Layout<_1>{},
                    group_modes<0, 2>(sB), group_modes<0, 2>(gB));

                int k_tile_count = size<1>(tAgA);

                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int qid = gq % QSIZE;

                    // Arrive-then-wait on empty barrier (producer counts in total)
                    EmptyBarrier::arrive(&shared.empty_barrier[qid]);
                    EmptyBarrier::wait(&shared.empty_barrier[qid], phase);

                    // Set expected TX bytes — each CTA's producer sets this
                    // for data arriving in its own smem (multicast or unicast).
                    FullBarrier::arrive_and_expect_tx(
                        &shared.full_barrier[qid], tma_tx_bytes);

                    // Load A tile (multicast if CLUSTER_N > 1)
                    if constexpr (CLUSTER_N > 1) {
                        if (rank_n == 0) {
                            copy(tma_a.with(shared.full_barrier[qid], a_mcast_mask),
                                 tAgA(_, k_tile), tAsA(_, qid));
                        }
                    } else {
                        copy(tma_a.with(shared.full_barrier[qid]),
                             tAgA(_, k_tile), tAsA(_, qid));
                    }

                    // Load B tile (multicast if CLUSTER_M > 1)
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

        // Producer tail: cluster sync to ensure no CTA exits while peers
        // are still using remote barriers.
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
        return;
    }

    // ==================================================================
    // CONSUMER (WG1 .. WG_NCS)
    // ==================================================================
    constexpr int NRG_CONS = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
    cutlass::arch::warpgroup_reg_alloc<NRG_CONS>();

    int consumer_thread = tid_in_wg;          // 0..127
    int csid            = warp_group_idx - 1; // 0..NCS-1

    // Initial empty signal: all slots are free for the producer.
    // One thread per consumer warpgroup arrives at every CTA's empty barrier.
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

    // Accumulator fragment (shape determined from WM × BN tile)
    Tensor mC_ref = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    Tensor gC_ref = local_tile(mC_ref, make_shape(Int<WM>{}, Int<BN>{}),
                               make_coord(0, 0));
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_ref));

    // Epilogue barrier: all consumer threads sync before/after writing C.
    cutlass::arch::NamedBarrier epi_barrier(NCS * 128,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    int tile_id;
    int gq_cons = 0, phase_cons = 0;
    while ((tile_id = schedule.next()) != -1) {
        int super_n = tile_id % (N / SUPER_BN);
        int super_m = tile_id / (N / SUPER_BN);
        int bm = (super_m * CLUSTER_M + rank_m) * BM;
        int bn = (super_n * CLUSTER_N + rank_n) * BN;

        // gC for this consumer's WM rows
        Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
        Tensor gC = local_tile(mC, make_shape(Int<WM>{}, Int<BN>{}),
                               make_coord(bm / WM + csid, bn / BN));
        Tensor tCgC = thr_mma.partition_C(gC);

        clear(tCrC);

        int k_tile_count = K / BK;

        #pragma unroll 1
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            int qid = gq_cons % QSIZE;

            // Wait for producer to fill this slot
            FullBarrier::wait(&shared.full_barrier[qid], phase_cons);

            // Per-consumer, per-stage 2-D smem sub-views
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

            // Signal empty: release this slot across all cluster CTAs
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

        // Epilogue: sync consumers, then write C
        epi_barrier.sync();
        axpby(alpha, tCrC, beta, tCgC);
        epi_barrier.sync();
    }

    // Consumer tail: cluster sync before exit
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}


// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 256, int BK = 64, int NCS = 2, int QSIZE = 4,
          int CLUSTER_M = 2, int CLUSTER_N = 1>
void hgemm_cluster(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    constexpr int WM       = 64;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;
    constexpr int NSM      = 128;
    static_assert(BM == WM * NCS, "BM must equal 64 * NCS");
    static_assert(NSM % NCLUSTER == 0, "NSM must be divisible by NCLUSTER");

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
    // TMA descriptors — multicast variant uses SM90_TMA_LOAD_MULTICAST
    // with cluster_size=1 so no smem partitioning; each CTA receives
    // the full tile (replicated multicast). Unicast uses SM90_TMA_LOAD.
    // ---------------------------------------------------------------
    Tensor mA_desc = make_tensor(A, make_shape(m, k), dA);
    Tensor mB_desc = make_tensor(B, make_shape(n, k), dB);

    // Multicast TMA: use cluster_size=1 so each CTA receives the FULL tile
    // (replicated multicast). The multicast mask in copy(...with(bar, mask))
    // handles multi-CTA delivery; the descriptor's box covers the full tile.
    auto tmaA = [&]() {
        if constexpr (CLUSTER_N > 1) {
            return make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mA_desc,
                                 sA_layout(_, _, 0),
                                 make_shape(Int<BM>{}, Int<BK>{}),
                                 Int<1>{});
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
                                 make_shape(Int<BN>{}, Int<BK>{}),
                                 Int<1>{});
        } else {
            return make_tma_atom(SM90_TMA_LOAD{}, mB_desc,
                                 sB_layout(_, _, 0),
                                 make_shape(Int<BN>{}, Int<BK>{}));
        }
    }();

    // ---------------------------------------------------------------
    // Launch config
    // ---------------------------------------------------------------
    constexpr int NUM_THREADS = (NCS + 1) * 128;
    constexpr int smem_bytes  = sizeof(HgemmClusterSharedStorage<BM, BN, BK, QSIZE>);

    auto kernel = hgemm_cluster_device<BM, BN, BK, NCS, QSIZE, NSM,
        CLUSTER_M, CLUSTER_N,
        decltype(tmaA), decltype(sA_layout),
        decltype(tmaB), decltype(sB_layout),
        decltype(sA_sub_layout), decltype(sB_sub_layout),
        decltype(dC), decltype(mma)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    // Persistent launch: exactly NSM CTAs
    dim3 block(NUM_THREADS);
    dim3 grid(NSM);

    kernel<<<grid, block, smem_bytes>>>(
        m, n, k, alpha,
        tmaA, sA_layout,
        tmaB, sB_layout,
        sA_sub_layout, sB_sub_layout,
        beta, C, dC, mma);
}
