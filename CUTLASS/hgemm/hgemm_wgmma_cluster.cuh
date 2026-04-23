#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

/*
 * Hopper TMA + WGMMA HGEMM with 1×2 Cluster: C = alpha * A^T * B + beta * C
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * Extends hgemm_wgmma.cuh with thread block cluster cooperation:
 *   - 1×2 cluster along N dimension (2 CTAs share the same M tile)
 *   - TMA multicast loads A once into both CTAs' shared memory
 *   - Each CTA independently loads its own B tile via regular TMA
 *   - Reduces global memory traffic for A by 2×
 *
 * Same warp-specialized design as hgemm_wgmma.cuh:
 *   WG0 (threads 0-127):   Producer — issues TMA async loads
 *   WG1 (threads 128-255): Consumer — does WGMMA SS compute
 */

// ---------------------------------------------------------------------------
// Shared storage
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_STAGES>
struct HgemmWgmmaClusterSharedStorage {
    alignas(128) cute::half_t smemA[NUM_STAGES * BM * BK];
    alignas(128) cute::half_t smemB[NUM_STAGES * BN * BK];
    alignas(8)   uint64_t     full_barrier[NUM_STAGES];
    alignas(8)   uint64_t     empty_barrier[NUM_STAGES];
};

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------
template <int BM_, int BN_, int BK_, int NUM_STAGES_,
          class TMA_A, class TMA_B,
          class TiledMMA, class SmemLayoutA, class SmemLayoutB,
          class CStride>
__global__ static
__launch_bounds__(256)
void hgemm_wgmma_cluster_device(
    int M, int N, int K,
    float alpha,
    CUTE_GRID_CONSTANT TMA_A const tma_load_a,
    CUTE_GRID_CONSTANT TMA_B const tma_load_b,
    float beta,
    cute::half_t* C, CStride dC,
    TiledMMA mma,
    SmemLayoutA sA_layout,
    SmemLayoutB sB_layout)
{
    using namespace cute;

    constexpr int BM = BM_;
    constexpr int BN = BN_;
    constexpr int BK = BK_;
    constexpr int NUM_STAGES = NUM_STAGES_;
    constexpr int WG_SIZE = 128;
    constexpr int CLUSTER_N = 2;

    int warp_group_idx = threadIdx.x / WG_SIZE;

    // Cluster coordination
    uint32_t block_rank = cute::block_rank_in_cluster();
    // 1×2 cluster along N: block_rank 0 and 1 share the same M tile
    // blockIdx.y already accounts for full CTA grid, each CTA handles its own N tile

    // ---------- Shared memory ----------
    using SharedStorage = HgemmWgmmaClusterSharedStorage<BM, BN, BK, NUM_STAGES>;
    extern __shared__ char smem_raw[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_raw);

    uint64_t* full_barrier  = shared.full_barrier;
    uint64_t* empty_barrier = shared.empty_barrier;

    // ---------- Initialize barriers ----------
    if (threadIdx.x == 0) {
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            // full_barrier: 1 software arrival (the elected TMA leader arrives)
            cutlass::arch::ClusterTransactionBarrier::init(&full_barrier[s], 1);
            // empty_barrier: CLUSTER_N arrivals — each CTA's consumer must
            // signal ALL CTAs in the cluster before any producer reuses the stage
            // (because multicast TMA writes to all CTAs' smem).
            cutlass::arch::ClusterBarrier::init(&empty_barrier[s], CLUSTER_N);
            // Pre-arrive CLUSTER_N times to mark all stages as initially "empty"
            for (int c = 0; c < CLUSTER_N; ++c) {
                cutlass::arch::ClusterBarrier::arrive(&empty_barrier[s]);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    // Cluster-wide sync to ensure all CTAs see initialized barriers
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    // ---------- Rank-3 smem tensors ----------
    Tensor sA = make_tensor(make_smem_ptr(shared.smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(shared.smemB), sB_layout);

    // ---------- TMA setup ----------
    Tensor mA_coord = tma_load_a.get_tma_tensor(make_shape(M, K));
    Tensor mB_coord = tma_load_b.get_tma_tensor(make_shape(N, K));

    // Each CTA computes its own tile: blockIdx.x for M, blockIdx.y for N
    // OOB CTAs (from grid rounding) clamp B to tile 0 and skip epilogue.
    bool valid_n = (int)blockIdx.y * BN < N;
    int b_tile = valid_n ? (int)blockIdx.y : 0;  // clamp OOB to valid tile

    Tensor gA = local_tile(mA_coord,
                           make_shape(Int<BM>{}, Int<BK>{}),
                           make_coord(blockIdx.x, _));
    Tensor gB = local_tile(mB_coord,
                           make_shape(Int<BN>{}, Int<BK>{}),
                           make_coord(b_tile, _));

    // TMA partitions: use cluster-aware slice for A (multicast)
    // For multicast A: get_slice uses the N-dimension cluster index
    auto cta_tma_a = tma_load_a.get_slice(block_rank);
    auto cta_tma_b = tma_load_b.get_slice(Int<0>{});

    Tensor tAgA = cta_tma_a.partition_S(gA);
    Tensor tAsA = cta_tma_a.partition_D(sA);
    Tensor tBgB = cta_tma_b.partition_S(gB);
    Tensor tBsB = cta_tma_b.partition_D(sB);

    int K_TILES = size<2>(gA);
    // Transaction bytes per CTA per stage: A (own multicast) + B (unicast)
    constexpr int tma_bytes = (BM * BK + BN * BK) * int(sizeof(half_t));

    // Multicast mask for A: broadcast to all CTAs in the 1×2 cluster
    uint16_t mcast_mask_a = 0;
    {
        auto block_layout = Layout<Shape<_1, Int<CLUSTER_N>>>{};
        for (int n = 0; n < CLUSTER_N; ++n) {
            mcast_mask_a |= (uint16_t(1) << block_layout(Int<0>{}, n));
        }
    }

    // ========================================================================
    // PRODUCER WARP GROUP (WG0): issue TMA loads
    // ========================================================================
    if (warp_group_idx == 0) {
        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            int stage = k_tile % NUM_STAGES;
            int phase = (k_tile / NUM_STAGES) % 2;

            cutlass::arch::ClusterBarrier::wait(&empty_barrier[stage], phase);

            if (threadIdx.x == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &full_barrier[stage], tma_bytes);

                // A: ALL CTAs issue multicast TMA. The HW multicasts data to
                // all CTAs in the mask, but transaction bytes only arrive on
                // the issuing CTA's barrier — so every CTA must issue its own.
                copy(tma_load_a.with(full_barrier[stage], mcast_mask_a),
                     tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
                // B: each CTA loads its own B tile (unicast)
                copy(tma_load_b.with(full_barrier[stage]),
                     tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
            }
        }

        // Producer tail: wait for consumers to release all stages before exiting.
        // This prevents early exit while other CTAs in cluster still work.
        {
            int final_k = K_TILES > 0 ? K_TILES - 1 : 0;
            CUTE_UNROLL
            for (int s = 0; s < NUM_STAGES; ++s) {
                if (s <= final_k) {
                    int phase = (final_k / NUM_STAGES) % 2;
                    cutlass::arch::ClusterBarrier::wait(&empty_barrier[s], phase);
                }
            }
        }
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    }
    // ========================================================================
    // CONSUMER WARP GROUP (WG1): WGMMA compute + epilogue
    // ========================================================================
    else {
        int consumer_thread = threadIdx.x - WG_SIZE;

        Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
        Tensor gC = local_tile(mC,
                               make_shape(Int<BM>{}, Int<BN>{}),
                               make_coord(blockIdx.x, valid_n ? (int)blockIdx.y : 0));

        auto thr_mma = mma.get_slice(consumer_thread);

        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);

        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCrC = thr_mma.make_fragment_C(tCgC);
        clear(tCrC);

        // ---------- Mainloop (all CTAs participate) ----------
        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            int stage = k_tile % NUM_STAGES;
            int phase = (k_tile / NUM_STAGES) % 2;

            cutlass::arch::ClusterBarrier::wait(&full_barrier[stage], phase);

            warpgroup_fence_operand(tCrC);
            warpgroup_arrive();

            gemm(mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrC);

            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrC);

            // Each CTA's consumer must arrive on ALL CTAs' empty barriers
            // so that every producer knows it's safe to reuse this stage.
            if (consumer_thread == 0) {
                cutlass::arch::ClusterBarrier::arrive(&empty_barrier[stage], block_rank, 1);
            }
            if (consumer_thread == 1) {
                uint32_t other_rank = 1 - block_rank;
                cutlass::arch::ClusterBarrier::arrive(&empty_barrier[stage], other_rank, 1);
            }
        }

        // ---------- Epilogue: only valid CTAs write ----------
        if (valid_n) {
            axpby(alpha, tCrC, beta, tCgC);
        }

        // Consumer tail: cluster sync before exit
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 64, int NUM_STAGES = 4>
void hgemm_wgmma_cluster(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    constexpr int CLUSTER_N = 2;

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // GMMA-compatible smem layout
    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<half_t>;

    auto sA_layout_tma = tile_to_shape(SmemLayoutAtom{},
                             make_shape(Int<BM>{}, Int<BK>{}));
    auto sB_layout_tma = tile_to_shape(SmemLayoutAtom{},
                             make_shape(Int<BN>{}, Int<BK>{}));

    auto sA_layout = tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<BM>{}, Int<BK>{}, Int<NUM_STAGES>{}));
    auto sB_layout = tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<BN>{}, Int<BK>{}, Int<NUM_STAGES>{}));

    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(m, k), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(n, k), dB);

    // TMA A: multicast across cluster N dimension
    auto tma_load_a = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, mA, sA_layout_tma,
                                     make_shape(Int<BM>{}, Int<BK>{}),
                                     Int<CLUSTER_N>{});
    // TMA B: unicast (each CTA loads its own B)
    auto tma_load_b = make_tma_copy(SM90_TMA_LOAD{}, mB, sB_layout_tma);

    auto mma_op = make_tiled_mma(
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}
    );

    constexpr int smem_bytes = int(sizeof(
        HgemmWgmmaClusterSharedStorage<BM, BN, BK, NUM_STAGES>));

    dim3 block(256);
    // Full CTA grid — grid.y must be a multiple of CLUSTER_N for valid cluster launch
    int grid_m = size(ceil_div(m, Int<BM>{}));
    int grid_n = size(ceil_div(n, Int<BN>{}));
    // Round up grid_n to multiple of CLUSTER_N
    grid_n = ((grid_n + CLUSTER_N - 1) / CLUSTER_N) * CLUSTER_N;
    dim3 grid(grid_m, grid_n);

    auto kernel = hgemm_wgmma_cluster_device<BM, BN, BK, NUM_STAGES,
        decltype(tma_load_a), decltype(tma_load_b),
        decltype(mma_op), decltype(sA_layout), decltype(sB_layout),
        decltype(dC)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    // Launch with cluster attributes via cudaLaunchKernelEx
    dim3 cluster_dims(1, CLUSTER_N, 1);

    void* kernel_params[] = {
        &m, &n, &k, &alpha,
        &tma_load_a,
        &tma_load_b,
        &beta,
        &C, &dC,
        &mma_op,
        &sA_layout,
        &sB_layout,
    };

    cutlass::ClusterLauncher::launch(
        grid, cluster_dims, block, smem_bytes, cudaStream_t{0},
        reinterpret_cast<void const*>(kernel),
        kernel_params);
}
