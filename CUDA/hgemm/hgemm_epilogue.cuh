#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cooperative_groups.h>
#include <mma.h>
#include <cassert>
#include "ptx_wrapper.cuh"
#include "scheduler.cuh"

/*
 * PERSISTENT CLUSTER-BASED HGEMM with TMA STORE EPILOGUE
 * C = alpha * A^T * B + beta * C   (TN col-major).
 *
 * Extends hgemm_cluster with a TMA bulk store epilogue.  Instead of
 * thousands of individual scalar writes to global memory, consumers
 * stage their fp32 accumulators → fp16 into a shared-memory buffer (sC),
 * then one thread issues a single TMA bulk store (smem → global) for
 * the entire BM×BN output tile.
 *
 * Epilogue flow per tile:
 *   1. Each consumer warpgroup writes its accumulator rows to sC (smem).
 *   2. Consumer-only barrier sync (epi) — all rows committed.
 *   3. Thread 0 issues: fence → TMA store → commit → wait_read.
 *   4. Consumer-only barrier sync (epi) — smem safe to reuse next tile.
 *
 * All other aspects (TMA multicast load, warp specialization, cluster,
 * persistent scheduling) are identical to hgemm_cluster.
 */

namespace hgemm_epilogue {

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

template <int BM, int BN, int BK, int QSIZE>
struct smem {
    alignas(128) half A[BM*BK*QSIZE];
    alignas(128) half B[BK*BN*QSIZE];
    alignas(128) half C[BM*BN];
};

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NSM,
         int CLUSTER_M, int CLUSTER_N>
__global__ __launch_bounds__(NUM_THREADS)
void __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
hgemm_epilogue_device(
    const __grid_constant__ CUtensorMap tmapA,
    const __grid_constant__ CUtensorMap tmapB,
    const __grid_constant__ CUtensorMap tmapC,
    int M, int N, int K,
    float alpha, float beta,
    half* __restrict__ C, int ldC)
{
    using namespace ptx_wrapper;
    namespace cg = cooperative_groups;

    constexpr int WM = 64;
    constexpr int WN = BN;
    constexpr int WK = 16;
    static_assert(BM % WM == 0 && BN % WN == 0);

    constexpr int NWG = NUM_THREADS / 128;
    constexpr int NCS = NWG - 1;
    constexpr int WGM = WM * NCS;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;

    // Cluster identity
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t cluster_id   = blockIdx.x / NCLUSTER;
    uint32_t cluster_rank = cluster.block_rank();
    uint32_t rank_m = cluster_rank / CLUSTER_N;
    uint32_t rank_n = cluster_rank % CLUSTER_N;

    // Shared memory
    extern __shared__ __align__(128) char smem_buf[];
    auto &s = *reinterpret_cast<smem<BM, BN, BK, QSIZE>*>(smem_buf);
    half* sA = reinterpret_cast<half*>(s.A);
    half* sB = reinterpret_cast<half*>(s.B);
    half* sC = reinterpret_cast<half*>(s.C);

    int wgid = threadIdx.x / 128;
    int csid = wgid - 1;
    int tid  = threadIdx.x % 128;

    // Barriers ----------------------------------------------------------------
    // full/empty: TMA load pipeline (same as hgemm_cluster)
    // epi: consumer-only sync for epilogue staging + TMA store
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[QSIZE], empty[QSIZE], epi;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            init(&full[i],  NCS * 128 + 1);
            init(&empty[i], NCS * NCLUSTER + 1);
        }
        init(&epi, NCS * 128);
        cde::fence_proxy_async_shared_cta();
    }
    // All CTAs must finish barrier init before any CTA accesses remote barriers.
    cluster.sync();

    // Scheduler: cluster-level super-blocks -----------------------------------
    constexpr int SUPER_BM = BM * CLUSTER_M;
    constexpr int SUPER_BN = BN * CLUSTER_N;
    scheduler::Schedule<1, NSM/NCLUSTER, SUPER_BM, SUPER_BN,
                        16/CLUSTER_M, 8/CLUSTER_N> schedule(M, N, cluster_id);

    // Multicast masks (constant for the entire kernel) ------------------------
    uint16_t b_mcast_mask = 0;
    if constexpr (CLUSTER_M > 1) {
        for (int i = 0; i < CLUSTER_M; ++i)
            b_mcast_mask |= uint16_t(1) << (i * CLUSTER_N + rank_n);
    }
    uint16_t a_mcast_mask = 0;
    if constexpr (CLUSTER_N > 1) {
        a_mcast_mask = uint16_t((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
    }

    constexpr int A_BYTES = BM * BK * (int)sizeof(half);
    constexpr int B_BYTES = BK * BN * (int)sizeof(half);

    if (wgid == 0) { // ==================== PRODUCER ====================
        constexpr int NRG = (NCS <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<NRG>();

        if (tid == 0) {
            int tile_id;
            while ((tile_id = schedule.next()) != -1) {
                // Decode super-block tile → per-CTA element offsets
                int super_n = tile_id % (N / SUPER_BN);
                int super_m = tile_id / (N / SUPER_BN);
                int bm = (super_m * CLUSTER_M + rank_m) * BM;
                int bn = (super_n * CLUSTER_N + rank_n) * BN;

                for (int bk = 0, qid = 0; bk < K; bk += BK, qid = (qid + 1) % QSIZE) {
                    // Wait for all cluster consumers to release this slot
                    empty[qid].wait(empty[qid].arrive());

                    // Set expected TX bytes — TMA completions (local + multicast) add up
                    barrier::arrival_token _ =
                        cuda::device::barrier_arrive_tx(full[qid], 1, A_BYTES + B_BYTES);

                    // Load A tile (unicast or multicast)
                    if constexpr (CLUSTER_N > 1) {
                        if (rank_n == 0)
                            cp_async_bulk_tensor_2d_multicast(
                                &sA[qid * BM * BK], &tmapA, bk, bm,
                                full[qid], a_mcast_mask);
                    } else {
                        cde::cp_async_bulk_tensor_2d_global_to_shared(
                            &sA[qid * BM * BK], &tmapA, bk, bm, full[qid]);
                    }

                    // Load B tile (multicast or unicast)
                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0)
                            cp_async_bulk_tensor_2d_multicast(
                                &sB[qid * BK * BN], &tmapB, bk, bn,
                                full[qid], b_mcast_mask);
                    } else {
                        cde::cp_async_bulk_tensor_2d_global_to_shared(
                            &sB[qid * BK * BN], &tmapB, bk, bn, full[qid]);
                    }
                }
            }
        }
    } else { // ==================== CONSUMER ====================
        constexpr int NRG = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
        warpgroup_reg_alloc<NRG>();

        // Initial empty signal: all slots are free for the producer.
        // One thread per consumer warpgroup arrives at every CTA's empty barrier.
        if (tid == 0) {
            for (int i = 0; i < QSIZE; ++i) {
                for (uint32_t c = 0; c < NCLUSTER; ++c) {
                    if (c == cluster_rank) {
                        (void)empty[i].arrive();
                    } else {
                        auto* remote = cluster.map_shared_rank(&empty[i], c);
                        arrive_remote_barrier(remote);
                    }
                }
            }
        }

        float d[BM/WGM][WN/16][8];

        int tile_id;
        while ((tile_id = schedule.next()) != -1) {
            int super_n = tile_id % (N / SUPER_BN);
            int super_m = tile_id / (N / SUPER_BN);
            int bm = (super_m * CLUSTER_M + rank_m) * BM;
            int bn = (super_n * CLUSTER_N + rank_n) * BN;
            half* C_ptr = C + bm + bn * M;

            // Zero accumulators
            #pragma unroll
            for (int i = 0; i < BM/WGM; i++)
                #pragma unroll
                for (int j = 0; j < WN/16; j++)
                    #pragma unroll
                    for (int k = 0; k < 8; k++)
                        d[i][j][k] = 0.f;

            for (int bk = 0, qid = 0; bk < K; bk += BK, qid = (qid + 1) % QSIZE) {
                // Wait for producer to fill this slot
                full[qid].wait(full[qid].arrive());

                // WGMMA compute
                warpgroup_arrive();
                #pragma unroll
                for (int wm = 0; wm < BM; wm += WGM) {
                    half* sAm = &sA[qid * BM * BK + (wm + csid * WM) * BK];
                    half* sBm = &sB[qid * BK * BN];
                    #pragma unroll
                    for (int wk = 0; wk < BK; wk += WK) {
                        wgmma<WN, 1, 1, 1, 0, 0>(d[wm/WGM], &sAm[wk], &sBm[wk]);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();

                // Signal empty: release this slot across all cluster CTAs
                if (tid == 0) {
                    for (uint32_t c = 0; c < NCLUSTER; ++c) {
                        if (c == cluster_rank) {
                            (void)empty[qid].arrive();
                        } else {
                            auto* remote = cluster.map_shared_rank(&empty[qid], c);
                            arrive_remote_barrier(remote);
                        }
                    }
                }
            }

            // === TMA STORE EPILOGUE ===
            // Step 1: Stage accumulators (fp32 → fp16) into sC (col-major BM×BN)
            {
                int t0 = tid % 4;
                int t1 = (tid / 4) % 8;
                int t2 = (tid / 32) % 4;
                int mo = t1 + 16*t2 + csid * WM;
                int no = 2*t0;
                #pragma unroll
                for (int wm = 0; wm < BM; wm += WGM) {
                    int m = mo + wm;
                    #pragma unroll
                    for (int w = 0; w < WN/16; w++) {
                        int n = no + 16*w;
                        #define SC(i, j)   sC[(i) + (j) * BM]
                        #define GC(i, j)   C_ptr[(i) + (j) * M]
                        #define WR(i, j, v) SC(i, j) = (half)(alpha * (v) + beta * (float)GC(i, j))
                        WR(m,   n,   d[wm/WGM][w][0]);
                        WR(m,   n+1, d[wm/WGM][w][1]);
                        WR(m+8, n,   d[wm/WGM][w][2]);
                        WR(m+8, n+1, d[wm/WGM][w][3]);
                        WR(m,   n+8, d[wm/WGM][w][4]);
                        WR(m,   n+9, d[wm/WGM][w][5]);
                        WR(m+8, n+8, d[wm/WGM][w][6]);
                        WR(m+8, n+9, d[wm/WGM][w][7]);
                        #undef WR
                        #undef GC
                        #undef SC
                    }
                }
            }

            // Step 2: Sync all consumers — sC is fully written
            epi.arrive_and_wait();

            // Step 3: One thread issues TMA bulk store: sC → global C
            if (csid == 0 && tid == 0) {
                cde::fence_proxy_async_shared_cta();
                cde::cp_async_bulk_tensor_2d_shared_to_global(&tmapC, bm, bn, sC);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group_read<0>();
            }

            // Step 4: Sync — smem sC is safe to reuse for next tile
            epi.arrive_and_wait();
        }
    }
}


template<int BM, int BN, int BK, int NUM_CONSUMERS = 2, int QSIZE = 3,
         int CLUSTER_M = 2, int CLUSTER_N = 1>
void hgemm_epilogue(
    int M, int N, int K,
    float alpha,
    const half* A, int ldA,
    const half* B, int ldB,
    float beta,
    half* C, int ldC)
{
    static_assert(NUM_CONSUMERS >= 1);
    constexpr int NUM_THREADS = (NUM_CONSUMERS + 1) * 128;
    constexpr int NUM_SM = 128;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;
    static_assert(NUM_SM % NCLUSTER == 0);

    CUtensorMap tmap_A{}, tmap_B{}, tmap_C{};
    (void) ptx_wrapper::build_tma_descriptor(&tmap_A, A, /*outer=*/M, /*inner=*/K, BK, BM);
    (void) ptx_wrapper::build_tma_descriptor(&tmap_B, B, /*outer=*/N, /*inner=*/K, BK, BN);
    // C descriptor: col-major (M×N), no swizzle — sC is written in plain col-major
    (void) ptx_wrapper::build_tma_descriptor(&tmap_C, C, /*outer=*/N, /*inner=*/M, BM, BN,
                                              CU_TENSOR_MAP_SWIZZLE_NONE);

    dim3 block(NUM_THREADS);
    dim3 grid(NUM_SM);

    auto kernel = hgemm_epilogue_device<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM,
                                       CLUSTER_M, CLUSTER_N>;

    constexpr int kSmemBytes = sizeof(smem<BM, BN, BK, QSIZE>);
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

    kernel<<<grid, block, kSmemBytes>>>(tmap_A, tmap_B, tmap_C,
                                         M, N, K, alpha, beta, C, ldC);
}

}  // namespace hgemm_epilogue
