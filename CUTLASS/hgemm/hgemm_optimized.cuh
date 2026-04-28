#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <vector>

/*
 * FULLY OPTIMIZED CuTe/CUTLASS HGEMM — 6 optimizations over hgemm_cluster:
 *
 *   1. Raw PTX mbarriers     — ClusterTransactionBarrier for full_bar,
 *                               ClusterBarrier for empty_bar with producer+consumer
 *                               arrival; only 1 thread arrives at full_bar.
 *   2. clear(tCrC) first K   — CuTe equivalent of scaleD=0 (skip explicit zero loop).
 *   3. Hilbert curve sched   — better L2 locality for output tile traversal.
 *   4. Async TMA store       — TMA store overlaps with next tile's K-loop;
 *                               wait_group_read deferred to next tile's epilogue.
 *   5. Swizzled 64-row sC    — 128B-swizzled col-major slices eliminate all
 *                               shared-memory bank conflicts in epilogue writes.
 *   6. Beta=0 fast path      — skip global C reads when beta==0.
 *
 * C = alpha * A^T * B + beta * C   (TN col-major).
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * Uses CuTe WGMMA (gemm) for the mainloop and manual WGMMA CLayout staging
 * for the swizzled epilogue.  TMA loads/stores use raw PTX with descriptors
 * built via cuTensorMapEncodeTiled.
 */

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
// Constants and helpers
// =========================================================================

constexpr int SPACE_LEN = 128;

// 128B swizzle for 64-row fp16 col-major slice.
// XOR element bits [5:3] with bits [8:6], matching CU_TENSOR_MAP_SWIZZLE_128B.
__device__ __forceinline__ int swz64(int m_local, int n) {
    int e = m_local + n * 64;
    return e ^ (((e >> 6) & 7) << 3);
}

// =========================================================================
// Shared memory layout
// =========================================================================

template <int BM, int BN, int BK, int QSIZE>
struct HgemmOptimizedSharedStorage {
    alignas(128) cute::half_t smemA[QSIZE * BM * BK];
    alignas(128) cute::half_t smemB[QSIZE * BN * BK];
    alignas(128) cute::half_t smemC[BM * BN];
    alignas(8)   uint64_t     full_bar[QSIZE];
    alignas(8)   uint64_t     empty_bar[QSIZE];
    int sched[SPACE_LEN];
};

// =========================================================================
// Raw PTX helpers for TMA load/store
// =========================================================================

namespace ptx_helpers {

__device__ __forceinline__ void tma_load_2d(
    void* dst, const CUtensorMap* tmap, uint64_t* bar, int c0, int c1)
{
    uint64_t tma_addr = reinterpret_cast<uint64_t>(tmap);
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(dst_addr), "l"(tma_addr), "r"(bar_addr),
           "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_2d_multicast(
    void* dst, const CUtensorMap* tmap, uint64_t* bar,
    int c0, int c1, uint16_t mask)
{
    uint64_t tma_addr = reinterpret_cast<uint64_t>(tmap);
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :: "r"(dst_addr), "l"(tma_addr), "r"(bar_addr),
           "r"(c0), "r"(c1), "h"(mask) : "memory");
}

__device__ __forceinline__ void tma_store_2d(
    const CUtensorMap* tmap, int c0, int c1, void* src)
{
    uint64_t tma_addr = reinterpret_cast<uint64_t>(tmap);
    uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3}], [%1];\n"
        :: "l"(tma_addr), "r"(src_addr), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory");
}

} // namespace ptx_helpers

// =========================================================================
// Host: build CUtensorMap for a 2D col-major half matrix tile.
// =========================================================================

inline CUresult build_tma_descriptor(
    CUtensorMap* tmap,
    const void* data,
    uint64_t outer_dim,
    uint64_t inner_dim,
    uint32_t box_inner,
    uint32_t box_outer,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B)
{
    CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    uint64_t size[2]        = { inner_dim, outer_dim };
    uint64_t stride[1]      = { inner_dim * sizeof(cute::half_t) };
    uint32_t box_size[2]    = { box_inner, box_outer };
    uint32_t elem_stride[2] = { 1, 1 };
    return cuTensorMapEncodeTiled(
        tmap, dtype, 2,
        const_cast<void*>(data), size, stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// =========================================================================
// Device kernel
// =========================================================================

template <int BM, int BN, int BK, int NCS, int QSIZE, int NSM,
          int CLUSTER_M, int CLUSTER_N>
__global__ static
__launch_bounds__((NCS + 1) * 128)
__cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
void hgemm_optimized_device(
    const __grid_constant__ CUtensorMap tmapA,
    const __grid_constant__ CUtensorMap tmapB,
    const __grid_constant__ CUtensorMap tmapC,
    int M, int N, int K,
    float alpha, float beta,
    cute::half_t* __restrict__ C, int ldC,
    const int* __restrict__ d_schedule)
{
    using namespace cute;
    using namespace ptx_helpers;
    namespace cg = cooperative_groups;

    constexpr int WM       = 64;
    constexpr int NCLUSTER = CLUSTER_M * CLUSTER_N;
    static_assert(BM == WM * NCS, "BM must equal 64 * NCS");

    // ------------------------------------------------------------------
    // Cluster identity
    // ------------------------------------------------------------------
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t cluster_id   = blockIdx.x / NCLUSTER;
    uint32_t cluster_rank = cluster.block_rank();
    uint32_t rank_m = cluster_rank / CLUSTER_N;
    uint32_t rank_n = cluster_rank % CLUSTER_N;

    // ------------------------------------------------------------------
    // Shared memory
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) char smem_raw[];
    auto& shared = *reinterpret_cast<
        HgemmOptimizedSharedStorage<BM, BN, BK, QSIZE>*>(smem_raw);

    int wgid = threadIdx.x / 128;
    int csid = wgid - 1;         // consumer id (0..NCS-1)
    int tid  = threadIdx.x % 128;

    // ------------------------------------------------------------------
    // Init raw mbarriers
    // ------------------------------------------------------------------
    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; i++) {
            // full_bar: only producer arrives (via expect_tx)
            cutlass::arch::ClusterTransactionBarrier::init(&shared.full_bar[i], 1);
            // empty_bar: NCS consumers (across cluster) + 1 producer
            cutlass::arch::ClusterBarrier::init(
                &shared.empty_bar[i], NCS * NCLUSTER + 1);
        }
        cutlass::arch::fence_barrier_init();
    }

    // ------------------------------------------------------------------
    // Load Hilbert schedule to smem
    // ------------------------------------------------------------------
    {
        constexpr int NUM_THREADS = (NCS + 1) * 128;
        const int* src = d_schedule + cluster_id * SPACE_LEN;
        for (int i = threadIdx.x; i < SPACE_LEN; i += NUM_THREADS)
            shared.sched[i] = src[i];
    }

    cluster.sync();

    // ------------------------------------------------------------------
    // Multicast masks (constant for entire kernel)
    // ------------------------------------------------------------------
    uint16_t b_mcast_mask = 0;
    if constexpr (CLUSTER_M > 1) {
        for (int i = 0; i < CLUSTER_M; ++i)
            b_mcast_mask |= uint16_t(1) << (i * CLUSTER_N + rank_n);
    }
    uint16_t a_mcast_mask = 0;
    if constexpr (CLUSTER_N > 1) {
        a_mcast_mask = uint16_t((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
    }

    constexpr int A_BYTES = BM * BK * (int)sizeof(cute::half_t);
    constexpr int B_BYTES = BK * BN * (int)sizeof(cute::half_t);

    // ------------------------------------------------------------------
    // WGMMA setup (smem layouts + tiled MMA)
    // ------------------------------------------------------------------
    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<cute::half_t>;
    auto sA_sub_layout = tile_to_shape(SmemLayoutAtom{},
        make_shape(Int<WM>{}, Int<BK>{}));
    auto sB_layout = tile_to_shape(SmemLayoutAtom{},
        make_shape(Int<BN>{}, Int<BK>{}));

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

    // ==================================================================
    // PRODUCER (WG0, 128 threads — only thread 0 issues TMAs)
    // ==================================================================
    if (wgid == 0) {
        constexpr int NRG = (NCS <= 2 ? 24 : 32);
        cutlass::arch::warpgroup_reg_dealloc<NRG>();

        if (tid == 0) {
            int gq = 0, p = 0;
            for (int sit = 0; sit < SPACE_LEN; sit++) {
                int packed = shared.sched[sit];
                if (packed == -1) break;
                int super_m = packed >> 16;
                int super_n = packed & 0xFFFF;
                int bm = (super_m * CLUSTER_M + rank_m) * BM;
                int bn = (super_n * CLUSTER_N + rank_n) * BN;

                for (int bk = 0; bk < K; bk += BK) {
                    int qi = gq % QSIZE;

                    // Wait for consumers to free this slot
                    cutlass::arch::ClusterBarrier::arrive(&shared.empty_bar[qi]);
                    cutlass::arch::ClusterBarrier::wait(&shared.empty_bar[qi], p);

                    // Set TX expectation (1 arrive + pending bytes)
                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                        &shared.full_bar[qi], A_BYTES + B_BYTES);

                    // Load A
                    if constexpr (CLUSTER_N > 1) {
                        if (rank_n == 0)
                            tma_load_2d_multicast(
                                &shared.smemA[qi * BM * BK], &tmapA,
                                &shared.full_bar[qi], bk, bm, a_mcast_mask);
                    } else {
                        tma_load_2d(
                            &shared.smemA[qi * BM * BK], &tmapA,
                            &shared.full_bar[qi], bk, bm);
                    }

                    // Load B
                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0)
                            tma_load_2d_multicast(
                                &shared.smemB[qi * BK * BN], &tmapB,
                                &shared.full_bar[qi], bk, bn, b_mcast_mask);
                    } else {
                        tma_load_2d(
                            &shared.smemB[qi * BK * BN], &tmapB,
                            &shared.full_bar[qi], bk, bn);
                    }

                    gq++;
                    if (gq % QSIZE == 0) p ^= 1;
                }
            }
        }

    } else {
        // ==============================================================
        // CONSUMER (WG1 .. WG_NCS)
        // ==============================================================
        constexpr int NRG = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
        cutlass::arch::warpgroup_reg_alloc<NRG>();

        int consumer_thread = tid;  // 0..127

        // Initial empty signal: all slots free for producer.
        // One thread per consumer WG arrives at every CTA's empty_bar.
        if (consumer_thread == 0) {
            for (int i = 0; i < QSIZE; ++i) {
                for (uint32_t c = 0; c < NCLUSTER; ++c) {
                    if (c == cluster_rank)
                        cutlass::arch::ClusterBarrier::arrive(
                            &shared.empty_bar[i]);
                    else
                        cutlass::arch::ClusterBarrier::arrive(
                            &shared.empty_bar[i], c, 1);
                }
            }
        }

        // Create accumulator fragment (WM=64 × BN)
        auto thr_mma = mma.get_slice(consumer_thread);
        auto sC_ref_layout = make_layout(
            make_shape(Int<WM>{}, Int<BN>{}),
            make_stride(Int<1>{}, Int<WM>{}));
        Tensor sC_ref = make_tensor(
            make_smem_ptr(shared.smemC), sC_ref_layout);
        Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(sC_ref));

        // WGMMA CLayout decomposition for manual epilogue staging
        int t0 = consumer_thread % 4;
        int t1 = (consumer_thread / 4) % 8;
        int t2 = (consumer_thread / 32) % 4;
        int m_local = t1 + 16 * t2;   // [0, 63] within 64-row slice
        int no = 2 * t0;

        int gq = 0, p = 0;

        for (int sit = 0; sit < SPACE_LEN; sit++) {
            int packed = shared.sched[sit];
            if (packed == -1) break;
            int super_m = packed >> 16;
            int super_n = packed & 0xFFFF;
            int bm = (super_m * CLUSTER_M + rank_m) * BM;
            int bn = (super_n * CLUSTER_N + rank_n) * BN;

            clear(tCrC);

            // ---- K-loop (WGMMA compute via CuTe) ----
            for (int bk = 0; bk < K; bk += BK) {
                int qi = gq % QSIZE;

                // Wait for producer to fill this slot (no arrive — count=1)
                cutlass::arch::ClusterTransactionBarrier::wait(
                    &shared.full_bar[qi], p);

                // Build per-stage smem sub-views for CuTe WGMMA
                Tensor sA_wg = make_tensor(
                    make_smem_ptr(&shared.smemA[qi * BM * BK + csid * WM * BK]),
                    sA_sub_layout);
                Tensor sB_tile = make_tensor(
                    make_smem_ptr(&shared.smemB[qi * BK * BN]),
                    sB_layout);

                auto tCsA = thr_mma.partition_A(sA_wg);
                auto tCsB = thr_mma.partition_B(sB_tile);

                warpgroup_fence_operand(tCrC);
                warpgroup_arrive();
                gemm(mma, tCsA, tCsB, tCrC);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                warpgroup_fence_operand(tCrC);

                // Release slot across cluster
                if (consumer_thread == 0) {
                    for (uint32_t c = 0; c < NCLUSTER; ++c) {
                        if (c == cluster_rank)
                            cutlass::arch::ClusterBarrier::arrive(
                                &shared.empty_bar[qi]);
                        else
                            cutlass::arch::ClusterBarrier::arrive(
                                &shared.empty_bar[qi], c, 1);
                    }
                }

                gq++;
                if (gq % QSIZE == 0) p ^= 1;
            }

            // ---- ASYNC TMA STORE EPILOGUE (swizzled 64-row slices) ----

            // Wait for prior tile's TMA store to finish reading sC
            if (sit > 0) {
                if (consumer_thread == 0)
                    cp_async_bulk_wait_group_read<0>();
                // Named barrier sync among all consumer threads
                asm volatile("bar.sync 1, %0;\n"
                    :: "r"((uint32_t)(NCS * 128)));
            }

            // Stage accumulators (fp32 → fp16) into 128B-swizzled sC.
            // Access tCrC linearly — element order matches WGMMA CLayout:
            //   acc[w*8 + f] where w=0..BN/16-1, f=0..7
            {
                constexpr int SLICE_ELEMS = 64 * BN;
                int slice_idx = csid;  // BM == WM*NCS → 1 slice per consumer
                cute::half_t* sC_s = shared.smemC + slice_idx * SLICE_ELEMS;

                float* acc = reinterpret_cast<float*>(&tCrC(0));
                int idx = 0;

                #pragma unroll
                for (int w = 0; w < BN / 16; w++) {
                    int n = no + 16 * w;

                    if (beta == 0.f) {
                        // Optimization #6: beta=0 fast path — skip global C reads
                        sC_s[swz64(m_local,     n)]     = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local,     n + 1)] = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local + 8, n)]     = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local + 8, n + 1)] = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local,     n + 8)] = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local,     n + 9)] = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local + 8, n + 8)] = cute::half_t(alpha * acc[idx++]);
                        sC_s[swz64(m_local + 8, n + 9)] = cute::half_t(alpha * acc[idx++]);
                    } else {
                        int gm = bm + slice_idx * 64;
                        cute::half_t* C_sl = C + gm + (size_t)bn * M;
                        #define GCL(r, c) C_sl[(r) + (size_t)(c) * M]
                        sC_s[swz64(m_local,     n)]     = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local,     n)));
                        sC_s[swz64(m_local,     n + 1)] = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local,     n + 1)));
                        sC_s[swz64(m_local + 8, n)]     = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local + 8, n)));
                        sC_s[swz64(m_local + 8, n + 1)] = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local + 8, n + 1)));
                        sC_s[swz64(m_local,     n + 8)] = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local,     n + 8)));
                        sC_s[swz64(m_local,     n + 9)] = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local,     n + 9)));
                        sC_s[swz64(m_local + 8, n + 8)] = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local + 8, n + 8)));
                        sC_s[swz64(m_local + 8, n + 9)] = cute::half_t(alpha * acc[idx++] + beta * float(GCL(m_local + 8, n + 9)));
                        #undef GCL
                    }
                }
            }

            // Sync: all consumers done writing sC
            asm volatile("bar.sync 1, %0;\n"
                :: "r"((uint32_t)(NCS * 128)));

            // Async TMA store: each consumer stores its 64-row slice
            if (consumer_thread == 0) {
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                int slice_idx = csid;
                tma_store_2d(&tmapC, bm + slice_idx * 64, bn,
                             shared.smemC + slice_idx * 64 * BN);
                cp_async_bulk_commit_group();
            }
        }

        // Wait for final TMA store
        if (consumer_thread == 0)
            cp_async_bulk_wait_group_read<0>();
    }
}


// =========================================================================
// Host launcher
// =========================================================================

template <int BM = 128, int BN = 256, int BK = 64, int NCS = 2, int QSIZE = 3,
          int CLUSTER_M = 2, int CLUSTER_N = 1>
void hgemm_optimized(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    static_assert(NCS >= 1);
    static_assert(BM == 64 * NCS, "BM must equal 64 * NCS");
    constexpr int NUM_THREADS = (NCS + 1) * 128;
    constexpr int NSM         = 128;
    constexpr int NCLUSTER    = CLUSTER_M * CLUSTER_N;
    static_assert(NSM % NCLUSTER == 0);

    constexpr int SUPER_BM = BM * CLUSTER_M;
    constexpr int SUPER_BN = BN * CLUSTER_N;
    int tiles_m      = m / SUPER_BM;
    int tiles_n      = n / SUPER_BN;
    int num_clusters = NSM / NCLUSTER;

    // Build TMA descriptors (A, B: 128B swizzle; C: 128B swizzle, 64-row tiles)
    CUtensorMap tmap_A{}, tmap_B{}, tmap_C{};
    (void)build_tma_descriptor(&tmap_A, A, /*outer=*/m, /*inner=*/k, BK, BM);
    (void)build_tma_descriptor(&tmap_B, B, /*outer=*/n, /*inner=*/k, BK, BN);
    (void)build_tma_descriptor(&tmap_C, C, /*outer=*/n, /*inner=*/m, 64, BN,
                                CU_TENSOR_MAP_SWIZZLE_128B);

    // Compute Hilbert schedule on host
    std::vector<int> h_schedule(num_clusters * SPACE_LEN, -1);
    compute_hilbert_schedule(tiles_m, tiles_n, num_clusters, SPACE_LEN,
                             h_schedule.data());

    // Upload to device (cached static allocation)
    static int* d_sched_ptr = nullptr;
    static size_t d_sched_bytes = 0;
    size_t needed = num_clusters * SPACE_LEN * sizeof(int);
    if (d_sched_bytes < needed) {
        if (d_sched_ptr) cudaFree(d_sched_ptr);
        cudaMalloc(&d_sched_ptr, needed);
        d_sched_bytes = needed;
    }
    cudaMemcpy(d_sched_ptr, h_schedule.data(), needed, cudaMemcpyHostToDevice);

    // Launch
    auto kernel = hgemm_optimized_device<BM, BN, BK, NCS, QSIZE, NSM,
                                          CLUSTER_M, CLUSTER_N>;
    constexpr int smem_bytes = (int)sizeof(
        HgemmOptimizedSharedStorage<BM, BN, BK, QSIZE>);
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dim3 block(NUM_THREADS);
    dim3 grid(NSM);
    kernel<<<grid, block, smem_bytes>>>(tmap_A, tmap_B, tmap_C,
                                         m, n, k, alpha, beta, C, ldC,
                                         d_sched_ptr);
}
