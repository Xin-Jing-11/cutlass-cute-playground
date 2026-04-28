#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>
#include <utility>

/*
 * WGMMA + TMA HGEMM (no warp specialization):
 * C = alpha * A^T * B + beta * C   (TN col-major).
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * FP16 in/out, FP32 accumulator. Single smem buffer (no pipelining).
 * All threads participate in both TMA wait and WGMMA compute.
 * NWG warpgroups, each handling TILES_M * WM rows of M where
 * TILES_M = BM / (NWG * WM).
 */

// Helpers for compile-time unrolling over M-tiles
template <size_t... Is, typename F>
CUTE_HOST_DEVICE void static_for_impl(std::index_sequence<Is...>, F&& f) {
    (f(cute::Int<Is>{}), ...);
}

template <int N, typename F>
CUTE_HOST_DEVICE void static_for(F&& f) {
    static_for_impl(std::make_index_sequence<N>{}, static_cast<F&&>(f));
}

template <size_t... Is, typename F>
CUTE_HOST_DEVICE auto make_ctuple_impl(std::index_sequence<Is...>, F&& f) {
    return cute::make_tuple(f(cute::Int<Is>{})...);
}

template <int N, typename F>
CUTE_HOST_DEVICE auto make_ctuple_for(F&& f) {
    return make_ctuple_impl(std::make_index_sequence<N>{}, static_cast<F&&>(f));
}

template <int BM, int BN, int BK>
struct HgemmWgmmaTmaSharedStorage {
    alignas(128) cute::half_t smemA[BM * BK];
    alignas(128) cute::half_t smemB[BN * BK];
    alignas(8)   uint64_t     full_barrier;
};

template <int BM_, int BN_, int BK_, int NWG_,
          class TMA_A, class TMA_B,
          class TiledMMA,
          class SmemLayoutA, class SmemLayoutB,
          class SmemLayoutA_sub,
          class CStride>
__global__ static
__launch_bounds__(NWG_ * 128)
void hgemm_wgmma_tma_device(
    int M, int N, int K,
    float alpha, float beta,
    CUTLASS_GRID_CONSTANT TMA_A const tma_load_a,
    CUTLASS_GRID_CONSTANT TMA_B const tma_load_b,
    SmemLayoutA sA_layout,
    SmemLayoutB sB_layout,
    SmemLayoutA_sub sA_sub_layout,
    TiledMMA mma,
    cute::half_t* C, CStride dC)
{
    using namespace cute;

    constexpr int BM  = BM_;
    constexpr int BN  = BN_;
    constexpr int BK  = BK_;
    constexpr int NWG = NWG_;
    constexpr int WM  = 64;
    constexpr int TILES_M = BM / (NWG * WM);
    static_assert(BM % (NWG * WM) == 0, "BM must be divisible by NWG * 64");

    extern __shared__ __align__(128) char smem_raw[];
    auto& shared = *reinterpret_cast<HgemmWgmmaTmaSharedStorage<BM, BN, BK>*>(smem_raw);

    int wgid      = threadIdx.x / 128;
    int tid_in_wg = threadIdx.x % 128;

    using FullBarrier = cutlass::arch::ClusterTransactionBarrier;

    // Barrier init
    if (threadIdx.x == 0) {
        FullBarrier::init(&shared.full_barrier, 1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // TMA global views
    Tensor mA = tma_load_a.get_tma_tensor(make_shape(M, K));   // (M, K)
    Tensor mB = tma_load_b.get_tma_tensor(make_shape(N, K));   // (N, K)

    // Tile the global tensors for this CTA
    Tensor gA = local_tile(mA, make_shape(Int<BM>{}, Int<BK>{}),
                           make_coord(blockIdx.x, _));          // (BM, BK, k_tiles)
    Tensor gB = local_tile(mB, make_shape(Int<BN>{}, Int<BK>{}),
                           make_coord(blockIdx.y, _));          // (BN, BK, k_tiles)

    // Smem tensors (2D, reused each iteration)
    Tensor sA = make_tensor(make_smem_ptr(shared.smemA), sA_layout);  // (BM, BK)
    Tensor sB = make_tensor(make_smem_ptr(shared.smemB), sB_layout);  // (BN, BK)

    // TMA partitioning on 2D smem
    auto [tAgA, tAsA] = tma_partition(tma_load_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_load_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sB), group_modes<0, 2>(gB));

    constexpr int tma_bytes = BM * BK * sizeof(cute::half_t)
                            + BN * BK * sizeof(cute::half_t);

    int K_TILES = size<1>(tAgA);

    // MMA setup: create gC partitions and accumulators for each M-tile
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    auto thr_mma = mma.get_slice(tid_in_wg);

    // Each warpgroup handles TILES_M consecutive 64-row tiles
    auto all_tCgC = make_ctuple_for<TILES_M>([&](auto mi) {
        int m_idx = blockIdx.x * (BM / WM) + wgid * TILES_M + int(mi);
        Tensor gC_mi = local_tile(mC, make_shape(Int<WM>{}, Int<BN>{}),
                                  make_coord(m_idx, blockIdx.y));
        return thr_mma.partition_C(gC_mi);
    });

    auto all_tCrC = make_ctuple_for<TILES_M>([&](auto mi) {
        auto tCrC = thr_mma.make_fragment_C(cute::get<decltype(mi)::value>(all_tCgC));
        clear(tCrC);
        return tCrC;
    });

    // Mainloop
    int phase = 0;
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
        // Thread 0 issues TMA
        if (threadIdx.x == 0) {
            FullBarrier::arrive_and_expect_tx(&shared.full_barrier, tma_bytes);
            copy(tma_load_a.with(shared.full_barrier), tAgA(_, k_tile), tAsA);
            copy(tma_load_b.with(shared.full_barrier), tBgB(_, k_tile), tBsB);
        }

        // All threads wait for TMA completion
        FullBarrier::wait(&shared.full_barrier, phase);
        phase ^= 1;

        // WGMMA compute for each M-tile
        Tensor sB_all = make_tensor(make_smem_ptr(shared.smemB), sB_layout);

        static_for<TILES_M>([&](auto mi) {
            constexpr int MI = decltype(mi)::value;
            auto& tCrC_mi = cute::get<MI>(all_tCrC);

            Tensor sA_wg = make_tensor(
                make_smem_ptr(shared.smemA + (wgid * TILES_M + MI) * WM * BK),
                sA_sub_layout);

            auto thr_mma_local = mma.get_slice(tid_in_wg);
            Tensor tCsA = thr_mma_local.partition_A(sA_wg);
            Tensor tCsB = thr_mma_local.partition_B(sB_all);

            warpgroup_fence_operand(tCrC_mi);
            warpgroup_arrive();
            gemm(mma, tCsA, tCsB, tCrC_mi);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrC_mi);
        });

        __syncthreads();
    }

    // Epilogue
    static_for<TILES_M>([&](auto mi) {
        constexpr int MI = decltype(mi)::value;
        axpby(alpha, cute::get<MI>(all_tCrC), beta, cute::get<MI>(all_tCgC));
    });
}


// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 64, int NWG = 1>
void hgemm_wgmma_tma(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    constexpr int WM = 64;
    static_assert(BM % WM == 0, "BM must be a multiple of 64");

    // TN layout strides
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // Swizzled smem layout for WGMMA
    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<cute::half_t>;

    auto sA_layout     = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto sB_layout     = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}));
    auto sA_sub_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<WM>{}, Int<BK>{}));

    // Select WGMMA atom based on BN
    auto mma = [&]() {
        if constexpr (BN <= 64) {
            return make_tiled_mma(
                SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else if constexpr (BN <= 128) {
            return make_tiled_mma(
                SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else if constexpr (BN <= 192) {
            return make_tiled_mma(
                SM90_64x192x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        } else {
            static_assert(BN <= 256, "BN must be <= 256");
            return make_tiled_mma(
                SM90_64x256x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
        }
    }();

    // TMA descriptors
    Tensor mA_desc = make_tensor(A, make_shape(m, k), dA);
    Tensor mB_desc = make_tensor(B, make_shape(n, k), dB);

    auto tma_load_a = make_tma_atom(SM90_TMA_LOAD{}, mA_desc, sA_layout,
                                    make_shape(Int<BM>{}, Int<BK>{}));
    auto tma_load_b = make_tma_atom(SM90_TMA_LOAD{}, mB_desc, sB_layout,
                                    make_shape(Int<BN>{}, Int<BK>{}));

    // Shared memory size
    constexpr int smem_bytes = (int)sizeof(HgemmWgmmaTmaSharedStorage<BM, BN, BK>);

    dim3 grid(size(ceil_div(m, Int<BM>{})),
              size(ceil_div(n, Int<BN>{})));
    dim3 block(NWG * 128);

    auto kernel = hgemm_wgmma_tma_device<BM, BN, BK, NWG,
        decltype(tma_load_a), decltype(tma_load_b),
        decltype(mma),
        decltype(sA_layout), decltype(sB_layout),
        decltype(sA_sub_layout),
        decltype(dC)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    kernel<<<grid, block, smem_bytes>>>(
        m, n, k, alpha, beta,
        tma_load_a, tma_load_b,
        sA_layout, sB_layout, sA_sub_layout,
        mma, C, dC);
}
