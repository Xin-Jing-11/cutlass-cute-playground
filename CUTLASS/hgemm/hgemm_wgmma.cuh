#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>

/*
 * Hopper TMA + WGMMA HGEMM: C = alpha * A^T * B + beta * C   (TN layout)
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * SM90 warp-specialized design:
 *   WG0 (threads 0-127):   Producer — issues TMA async loads
 *   WG1 (threads 128-255): Consumer — does WGMMA SS compute
 *
 * Uses SM90_TMA_LOAD for gmem→smem, SM90_64x128x16_F32F16F16_SS for
 * warp-group MMA (both operands from smem), multistage pipeline with
 * mbarrier (ClusterTransactionBarrier) synchronization.
 *
 * Template params: BM, BN, BK (CTA tile), NUM_STAGES (pipeline depth).
 */

// ---------------------------------------------------------------------------
// Shared storage with proper alignment for barriers
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_STAGES>
struct HgemmWgmmaSharedStorage {
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
void hgemm_wgmma_device(
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

    int warp_group_idx = threadIdx.x / WG_SIZE;

    // ---------- Shared memory ----------
    using SharedStorage = HgemmWgmmaSharedStorage<BM, BN, BK, NUM_STAGES>;
    extern __shared__ char smem_raw[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_raw);

    uint64_t* full_barrier  = shared.full_barrier;
    uint64_t* empty_barrier = shared.empty_barrier;

    // ---------- Initialize barriers ----------
    if (threadIdx.x == 0) {
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            cutlass::arch::ClusterTransactionBarrier::init(&full_barrier[s], 1);
            cutlass::arch::ClusterBarrier::init(&empty_barrier[s], 1);
            // Pre-arrive: mark all stages as initially "empty" (phase 0 completes)
            cutlass::arch::ClusterBarrier::arrive(&empty_barrier[s]);
        }
    }
    __syncthreads();

    // ---------- Rank-3 smem tensors covering all stages ----------
    Tensor sA = make_tensor(make_smem_ptr(shared.smemA), sA_layout); // (BM, BK, STAGES)
    Tensor sB = make_tensor(make_smem_ptr(shared.smemB), sB_layout); // (BN, BK, STAGES)

    // ---------- TMA setup ----------
    Tensor mA_coord = tma_load_a.get_tma_tensor(make_shape(M, K));
    Tensor mB_coord = tma_load_b.get_tma_tensor(make_shape(N, K));

    Tensor gA = local_tile(mA_coord,
                           make_shape(Int<BM>{}, Int<BK>{}),
                           make_coord(blockIdx.x, _));   // (BM, BK, k_tiles)
    Tensor gB = local_tile(mB_coord,
                           make_shape(Int<BN>{}, Int<BK>{}),
                           make_coord(blockIdx.y, _));   // (BN, BK, k_tiles)

    auto cta_tma_a = tma_load_a.get_slice(Int<0>{});
    auto cta_tma_b = tma_load_b.get_slice(Int<0>{});

    // TMA partitions on rank-3 tensors → rank-4
    Tensor tAgA = cta_tma_a.partition_S(gA);   // (TMA, TMA_M, TMA_K, k_tiles)
    Tensor tAsA = cta_tma_a.partition_D(sA);   // (TMA, TMA_M, TMA_K, STAGES)
    Tensor tBgB = cta_tma_b.partition_S(gB);
    Tensor tBsB = cta_tma_b.partition_D(sB);

    int K_TILES = size<2>(gA);
    constexpr int tma_bytes = (BM * BK + BN * BK) * int(sizeof(half_t));

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
                // Index 4th mode: gmem by k_tile, smem by stage
                copy(tma_load_a.with(full_barrier[stage]),
                     tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
                copy(tma_load_b.with(full_barrier[stage]),
                     tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
            }
        }
    }
    // ========================================================================
    // CONSUMER WARP GROUP (WG1): WGMMA compute + epilogue
    // ========================================================================
    else {
        int consumer_thread = threadIdx.x - WG_SIZE;

        Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
        Tensor gC = local_tile(mC,
                               make_shape(Int<BM>{}, Int<BN>{}),
                               make_coord(blockIdx.x, blockIdx.y));

        auto thr_mma = mma.get_slice(consumer_thread);

        // MMA partitions on rank-3 smem → rank-4
        Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K, STAGES)
        Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K, STAGES)

        // Create GMMA descriptors from smem partitions (required for SS mode)
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCrC = thr_mma.make_fragment_C(tCgC);
        clear(tCrC);

        // ---------- Mainloop ----------
        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            int stage = k_tile % NUM_STAGES;
            int phase = (k_tile / NUM_STAGES) % 2;

            cutlass::arch::ClusterBarrier::wait(&full_barrier[stage], phase);

            warpgroup_fence_operand(tCrC);
            warpgroup_arrive();

            // WGMMA compute: gemm iterates over MMA_K blocks internally
            gemm(mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrC);

            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrC);

            // Signal stage free — only consumer leader thread
            if (consumer_thread == 0) {
                cutlass::arch::ClusterBarrier::arrive(&empty_barrier[stage]);
            }
        }

        // ---------- Epilogue: C = alpha * acc + beta * C ----------
        axpby(alpha, tCrC, beta, tCgC);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 64, int NUM_STAGES = 4>
void hgemm_wgmma(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // GMMA-compatible smem layout: K-major with 128-byte swizzle
    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<half_t>;

    // 2D layout for TMA descriptor creation (one stage)
    auto sA_layout_tma = tile_to_shape(SmemLayoutAtom{},
                             make_shape(Int<BM>{}, Int<BK>{}));
    auto sB_layout_tma = tile_to_shape(SmemLayoutAtom{},
                             make_shape(Int<BN>{}, Int<BK>{}));

    // Rank-3 smem layouts: (M/N, K, STAGES)
    auto sA_layout = tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<BM>{}, Int<BK>{}, Int<NUM_STAGES>{}));
    auto sB_layout = tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<BN>{}, Int<BK>{}, Int<NUM_STAGES>{}));

    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(m, k), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(n, k), dB);

    // TMA descriptors use 2D single-stage layout
    auto tma_load_a = make_tma_copy(SM90_TMA_LOAD{}, mA, sA_layout_tma);
    auto tma_load_b = make_tma_copy(SM90_TMA_LOAD{}, mB, sB_layout_tma);

    auto mma = make_tiled_mma(
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}
    );

    constexpr int smem_bytes = int(sizeof(
        HgemmWgmmaSharedStorage<BM, BN, BK, NUM_STAGES>));

    dim3 block(256);
    dim3 grid(size(ceil_div(m, Int<BM>{})),
              size(ceil_div(n, Int<BN>{})));

    auto kernel = hgemm_wgmma_device<BM, BN, BK, NUM_STAGES,
        decltype(tma_load_a), decltype(tma_load_b),
        decltype(mma), decltype(sA_layout), decltype(sB_layout),
        decltype(dC)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    kernel<<<grid, block, smem_bytes>>>(
        m, n, k, alpha,
        tma_load_a, tma_load_b,
        beta, C, dC, mma,
        sA_layout, sB_layout);
}
