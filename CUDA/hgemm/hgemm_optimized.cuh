#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <vector>
#include "ptx_wrapper.cuh"

/*
 * FULLY OPTIMIZED HGEMM — 6 optimizations over hgemm_epilogue:
 *
 *   1. Raw PTX mbarriers     — fewer synchronization tokens, only 1 thread
 *                               arrives at full_bar (vs NCS*128 with cuda::barrier).
 *   2. ScaleD=0 first K iter — skip explicit accumulator zeroing.
 *   3. Hilbert curve sched   — better L2 locality for output tile traversal.
 *   4. Async TMA store       — TMA store overlaps with next tile's K-loop;
 *                               wait_group_read deferred to next tile's epilogue.
 *   5. Swizzled 64-row sC    — 128B-swizzled col-major slices eliminate all
 *                               shared-memory bank conflicts in epilogue writes.
 *   6. Beta=0 fast path      — skip global C reads when beta==0.
 *
 * C = alpha * A^T * B + beta * C   (TN col-major).
 */

namespace hgemm_optimized {

// =========================================================================
// Host-side Hilbert curve scheduling
// =========================================================================

// Convert Hilbert distance d to (x,y) on n×n grid (n must be power-of-2).
static void d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        if (ry == 0) {
            if (rx == 1) { *x = s - 1 - *x; *y = s - 1 - *y; }
            int tmp = *x; *x = *y; *y = tmp;
        }
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

// Distribute Hilbert-ordered super-tiles round-robin to clusters.
// Output: schedule[cluster * SPACE_LEN + i] = (super_m << 16) | super_n.
static void compute_hilbert_schedule(
    int tiles_m, int tiles_n, int num_clusters, int SPACE_LEN, int* schedule)
{
    int hN = 1;
    while (hN < tiles_m || hN < tiles_n) hN *= 2;

    std::vector<int> ordered;
    ordered.reserve(tiles_m * tiles_n);
    for (int d = 0; d < hN * hN; d++) {
        int x, y;
        d2xy(hN, d, &x, &y);
        if (x < tiles_m && y < tiles_n)
            ordered.push_back((x << 16) | y);
    }

    for (int i = 0; i < num_clusters * SPACE_LEN; i++)
        schedule[i] = -1;

    std::vector<int> counts(num_clusters, 0);
    for (int i = 0; i < (int)ordered.size(); i++) {
        int c = i % num_clusters;
        if (counts[c] < SPACE_LEN)
            schedule[c * SPACE_LEN + counts[c]++] = ordered[i];
    }
}

// =========================================================================
// Shared memory layout
// =========================================================================

constexpr int SPACE_LEN = 128;

// 128B swizzle for 64-row fp16 col-major slice.
// XOR element bits [5:3] with bits [8:6], matching CU_TENSOR_MAP_SWIZZLE_128B.
__device__ __forceinline__ int swz64(int m_local, int n) {
    int e = m_local + n * 64;
    return e ^ (((e >> 6) & 7) << 3);
}

template <int BM, int BN, int BK, int QSIZE>
struct smem_t {
    alignas(128) half A[BM*BK*QSIZE];
    alignas(128) half B[BK*BN*QSIZE];
    alignas(128) half C[BM*BN];
    alignas(8) uint64_t full_bar[QSIZE];
    alignas(8) uint64_t empty_bar[QSIZE];
    int sched[SPACE_LEN];
};

// =========================================================================
// Device kernel
// =========================================================================

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NSM,
         int CLUSTER_M, int CLUSTER_N>
__global__ __launch_bounds__(NUM_THREADS)
void __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
hgemm_optimized_device(
    const __grid_constant__ CUtensorMap tmapA,
    const __grid_constant__ CUtensorMap tmapB,
    const __grid_constant__ CUtensorMap tmapC,
    int M, int N, int K,
    float alpha, float beta,
    half* __restrict__ C, int ldC,
    const int* __restrict__ d_schedule)
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
    auto &s = *reinterpret_cast<smem_t<BM, BN, BK, QSIZE>*>(smem_buf);
    half*     sA        = s.A;
    half*     sB        = s.B;
    half*     sC        = s.C;
    uint64_t* full_bar  = s.full_bar;
    uint64_t* empty_bar = s.empty_bar;
    int*      sched     = s.sched;

    int wgid = threadIdx.x / 128;
    int csid = wgid - 1;
    int tid  = threadIdx.x % 128;

    // ---- Init raw mbarriers ----
    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            mbar_init(&full_bar[i], 1);                   // producer only (expect_tx)
            mbar_init(&empty_bar[i], NCS * NCLUSTER + 1); // consumers + producer
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }

    // ---- Load Hilbert schedule to smem ----
    {
        const int* src = d_schedule + cluster_id * SPACE_LEN;
        for (int i = threadIdx.x; i < SPACE_LEN; i += NUM_THREADS)
            sched[i] = src[i];
    }

    cluster.sync();

    // ---- Multicast masks ----
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

    if (wgid == 0) {
        // ==================== PRODUCER ====================
        constexpr int NRG = (NCS <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<NRG>();

        if (tid == 0) {
            int gq = 0, p = 0;
            for (int sit = 0; sit < SPACE_LEN; sit++) {
                int packed = sched[sit];
                if (packed == -1) break;
                int super_m = packed >> 16;
                int super_n = packed & 0xFFFF;
                int bm = (super_m * CLUSTER_M + rank_m) * BM;
                int bn = (super_n * CLUSTER_N + rank_n) * BN;

                for (int bk = 0; bk < K; bk += BK) {
                    int qi = gq % QSIZE;

                    // Wait for consumers to free this slot
                    mbar_arrive(&empty_bar[qi]);
                    mbar_wait(&empty_bar[qi], p);

                    // Set TX expectation (1 arrive + pending bytes)
                    mbar_expect_tx(&full_bar[qi], A_BYTES + B_BYTES);

                    // Load A
                    if constexpr (CLUSTER_N > 1) {
                        if (rank_n == 0)
                            tma_load_2d_multicast_raw(
                                &sA[qi * BM * BK], &tmapA, &full_bar[qi],
                                bk, bm, a_mcast_mask);
                    } else {
                        tma_load_2d_raw(
                            &sA[qi * BM * BK], &tmapA, &full_bar[qi], bk, bm);
                    }

                    // Load B
                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0)
                            tma_load_2d_multicast_raw(
                                &sB[qi * BK * BN], &tmapB, &full_bar[qi],
                                bk, bn, b_mcast_mask);
                    } else {
                        tma_load_2d_raw(
                            &sB[qi * BK * BN], &tmapB, &full_bar[qi], bk, bn);
                    }

                    gq++;
                    if (gq % QSIZE == 0) p ^= 1;
                }
            }
        }
    } else {
        // ==================== CONSUMER ====================
        constexpr int NRG = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
        warpgroup_reg_alloc<NRG>();

        // Initial empty signal: free all pipeline slots
        if (tid == 0) {
            for (int i = 0; i < QSIZE; ++i) {
                for (uint32_t c = 0; c < NCLUSTER; ++c) {
                    if (c == cluster_rank)
                        mbar_arrive(&empty_bar[i]);
                    else
                        mbar_arrive_cluster(&empty_bar[i], c);
                }
            }
        }

        float d[BM/WGM][WN/16][8];
        int gq = 0, p = 0;

        for (int sit = 0; sit < SPACE_LEN; sit++) {
            int packed = sched[sit];
            if (packed == -1) break;
            int super_m = packed >> 16;
            int super_n = packed & 0xFFFF;
            int bm = (super_m * CLUSTER_M + rank_m) * BM;
            int bn = (super_n * CLUSTER_N + rank_n) * BN;

            // ---- K-loop (WGMMA compute) ----
            for (int bk = 0; bk < K; bk += BK) {
                int qi = gq % QSIZE;

                // Wait for producer to fill this slot (no arrive — count=1)
                mbar_wait(&full_bar[qi], p);

                // WGMMA compute
                warpgroup_arrive();
                if (bk == 0) {
                    // First K chunk: scaleD=0 overwrites accumulators
                    #pragma unroll
                    for (int wm = 0; wm < BM; wm += WGM) {
                        half* sAm = &sA[qi * BM * BK + (wm + csid * WM) * BK];
                        half* sBm = &sB[qi * BK * BN];
                        wgmma<WN, 0, 1, 1, 0, 0>(d[wm/WGM], &sAm[0], &sBm[0]);
                        #pragma unroll
                        for (int wk = WK; wk < BK; wk += WK)
                            wgmma<WN, 1, 1, 1, 0, 0>(d[wm/WGM], &sAm[wk], &sBm[wk]);
                    }
                } else {
                    #pragma unroll
                    for (int wm = 0; wm < BM; wm += WGM) {
                        half* sAm = &sA[qi * BM * BK + (wm + csid * WM) * BK];
                        half* sBm = &sB[qi * BK * BN];
                        #pragma unroll
                        for (int wk = 0; wk < BK; wk += WK)
                            wgmma<WN, 1, 1, 1, 0, 0>(d[wm/WGM], &sAm[wk], &sBm[wk]);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();

                // Release slot across cluster
                if (tid == 0) {
                    for (uint32_t c = 0; c < NCLUSTER; ++c) {
                        if (c == cluster_rank)
                            mbar_arrive(&empty_bar[qi]);
                        else
                            mbar_arrive_cluster(&empty_bar[qi], c);
                    }
                }

                gq++;
                if (gq % QSIZE == 0) p ^= 1;
            }

            // ---- ASYNC TMA STORE EPILOGUE (swizzled 64-row slices) ----

            // Wait for prior tile's TMA store to finish reading sC
            if (sit > 0) {
                if (tid == 0) cp_async_bulk_wait_group_read<0>();
                asm volatile("bar.sync 1, %0;\n" :: "r"((uint32_t)(NCS * 128)));
            }

            // Stage accumulators (fp32 → fp16) into 128B-swizzled 64-row slices
            {
                constexpr int SLICE_ELEMS = 64 * BN;
                int t0 = tid % 4;
                int t1 = (tid / 4) % 8;
                int t2 = (tid / 32) % 4;
                int m_local = t1 + 16*t2;   // [0, 63] within slice
                int no = 2*t0;

                #pragma unroll
                for (int wm = 0; wm < BM; wm += WGM) {
                    int slice_idx = (csid * WM + wm) / 64;
                    half* sC_s = sC + slice_idx * SLICE_ELEMS;

                    #pragma unroll
                    for (int w = 0; w < WN/16; w++) {
                        int n = no + 16*w;

                        if (beta == 0.f) {
                            sC_s[swz64(m_local,   n)]   = (half)(alpha * d[wm/WGM][w][0]);
                            sC_s[swz64(m_local,   n+1)] = (half)(alpha * d[wm/WGM][w][1]);
                            sC_s[swz64(m_local+8, n)]   = (half)(alpha * d[wm/WGM][w][2]);
                            sC_s[swz64(m_local+8, n+1)] = (half)(alpha * d[wm/WGM][w][3]);
                            sC_s[swz64(m_local,   n+8)] = (half)(alpha * d[wm/WGM][w][4]);
                            sC_s[swz64(m_local,   n+9)] = (half)(alpha * d[wm/WGM][w][5]);
                            sC_s[swz64(m_local+8, n+8)] = (half)(alpha * d[wm/WGM][w][6]);
                            sC_s[swz64(m_local+8, n+9)] = (half)(alpha * d[wm/WGM][w][7]);
                        } else {
                            int gm = bm + slice_idx * 64;
                            half* C_sl = C + gm + bn * M;
                            #define GCL(r, c) C_sl[(r) + (c) * M]
                            sC_s[swz64(m_local,   n)]   = (half)(alpha * d[wm/WGM][w][0] + beta * (float)GCL(m_local,   n));
                            sC_s[swz64(m_local,   n+1)] = (half)(alpha * d[wm/WGM][w][1] + beta * (float)GCL(m_local,   n+1));
                            sC_s[swz64(m_local+8, n)]   = (half)(alpha * d[wm/WGM][w][2] + beta * (float)GCL(m_local+8, n));
                            sC_s[swz64(m_local+8, n+1)] = (half)(alpha * d[wm/WGM][w][3] + beta * (float)GCL(m_local+8, n+1));
                            sC_s[swz64(m_local,   n+8)] = (half)(alpha * d[wm/WGM][w][4] + beta * (float)GCL(m_local,   n+8));
                            sC_s[swz64(m_local,   n+9)] = (half)(alpha * d[wm/WGM][w][5] + beta * (float)GCL(m_local,   n+9));
                            sC_s[swz64(m_local+8, n+8)] = (half)(alpha * d[wm/WGM][w][6] + beta * (float)GCL(m_local+8, n+8));
                            sC_s[swz64(m_local+8, n+9)] = (half)(alpha * d[wm/WGM][w][7] + beta * (float)GCL(m_local+8, n+9));
                            #undef GCL
                        }
                    }
                }
            }

            // Sync: all consumers done writing sC
            asm volatile("bar.sync 1, %0;\n" :: "r"((uint32_t)(NCS * 128)));

            // Async TMA store: each warpgroup stores its own 64-row slice(s)
            if (tid == 0) {
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                #pragma unroll
                for (int wm = 0; wm < BM; wm += WGM) {
                    int slice_idx = (csid * WM + wm) / 64;
                    tma_store_2d(&tmapC, bm + slice_idx * 64, bn,
                                 sC + slice_idx * 64 * BN);
                }
                cp_async_bulk_commit_group();
            }
        }

        // Wait for final TMA store
        if (tid == 0) cp_async_bulk_wait_group_read<0>();
    }
}

// =========================================================================
// Host launcher
// =========================================================================

template<int BM, int BN, int BK, int NUM_CONSUMERS = 2, int QSIZE = 3,
         int CLUSTER_M = 2, int CLUSTER_N = 1>
void hgemm_optimized(
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

    constexpr int SUPER_BM = BM * CLUSTER_M;
    constexpr int SUPER_BN = BN * CLUSTER_N;
    int tiles_m = M / SUPER_BM;
    int tiles_n = N / SUPER_BN;
    int num_clusters = NUM_SM / NCLUSTER;

    // Build TMA descriptors
    CUtensorMap tmap_A{}, tmap_B{}, tmap_C{};
    (void) ptx_wrapper::build_tma_descriptor(&tmap_A, A, M, K, BK, BM);
    (void) ptx_wrapper::build_tma_descriptor(&tmap_B, B, N, K, BK, BN);
    (void) ptx_wrapper::build_tma_descriptor(&tmap_C, C, N, M, 64, BN,
                                              CU_TENSOR_MAP_SWIZZLE_128B);

    // Compute Hilbert schedule on host
    std::vector<int> h_schedule(num_clusters * SPACE_LEN, -1);
    compute_hilbert_schedule(tiles_m, tiles_n, num_clusters, SPACE_LEN,
                             h_schedule.data());

    // Upload to device (cached static allocation)
    static int* d_schedule = nullptr;
    static size_t d_sched_bytes = 0;
    size_t needed = num_clusters * SPACE_LEN * sizeof(int);
    if (d_sched_bytes < needed) {
        if (d_schedule) cudaFree(d_schedule);
        cudaMalloc(&d_schedule, needed);
        d_sched_bytes = needed;
    }
    cudaMemcpy(d_schedule, h_schedule.data(), needed, cudaMemcpyHostToDevice);

    // Launch
    auto kernel = hgemm_optimized_device<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM,
                                          CLUSTER_M, CLUSTER_N>;
    constexpr int kSmemBytes = sizeof(smem_t<BM, BN, BK, QSIZE>);
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

    dim3 block(NUM_THREADS);
    dim3 grid(NUM_SM);
    kernel<<<grid, block, kSmemBytes>>>(tmap_A, tmap_B, tmap_C,
                                         M, N, K, alpha, beta, C, ldC,
                                         d_schedule);
}

}  // namespace hgemm_optimized
