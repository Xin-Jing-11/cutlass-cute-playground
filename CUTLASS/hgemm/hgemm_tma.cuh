#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>               // ldmatrix
#include <cute/arch/copy_sm90_tma.hpp>           // SM90_TMA_LOAD
#include <cute/atom/copy_traits_sm90_tma.hpp>    // make_tma_atom, tma_partition
#include <cute/atom/mma_traits_sm90_gmma.hpp>    // GMMA::Layout_K_SW128_Atom
#include <cutlass/arch/barrier.h>                // ClusterBarrier, ClusterTransactionBarrier
#include <cutlass/device_kernel.h>               // CUTLASS_GRID_CONSTANT

/*
 * Warp-specialized HGEMM: C = alpha * A^T * B + beta * C   (TN layout)
 * A(M,K):(K,1), B(N,K):(K,1), C(M,N):(1,M).
 *
 * 12 warps = 384 threads split into 3 warp groups:
 *   - WG0 (128 threads, warps 0..3):  TMA producer. A single elect_one thread
 *                                     issues cp.async.bulk.tensor for A and B
 *                                     into the next pipe stage.
 *   - WG1+WG2 (256 threads, warps 4..11): MMA consumers. 2×4 warp tiling over
 *                                         SM80_16x8x16_F32F16F16F32_TN atoms.
 *                                         Uses ldmatrix s2r + mma.sync.
 *
 * Producer/consumer synchronization is done with SM90 mbarriers:
 *   - full_barrier[pipe]  (ClusterTransactionBarrier): producer→consumer,
 *       TMA arrive_and_expect_tx + HW transaction-byte completion on full tile.
 *   - empty_barrier[pipe] (ClusterBarrier):            consumer→producer,
 *       each of the 256 consumer threads arrives when it has finished
 *       reading its share of the pipe stage.
 *
 * Runs on SM120 (Blackwell consumer) when compiled for sm_120a: the SM90_TMA_LOAD
 * atom emits the `shared::cta` PTX variant (see cute::config.hpp
 * CUTE_ARCH_TMA_SM120_ENABLED) and all mbarrier asms use `shared::cta` scope.
 */

template <int NUM_STAGES, int NUM_CONSUMER_THREADS, int NUM_PRODUCER_THREADS,
    class ProblemShape, class CtaTiler,
    class TmaA, class ASmemLayout,
    class TmaB, class BSmemLayout,
    class CStride, class TiledMMA,
    class S2RCopyA, class S2RCopyB>
__global__ static
__launch_bounds__(NUM_CONSUMER_THREADS + NUM_PRODUCER_THREADS)
void hgemm_tma_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    float alpha,
    CUTLASS_GRID_CONSTANT TmaA const tma_a, ASmemLayout sA_layout,
    CUTLASS_GRID_CONSTANT TmaB const tma_b, BSmemLayout sB_layout,
    float beta,
    cute::half_t* C, CStride dC,
    TiledMMA mma,
    S2RCopyA s2r_A, S2RCopyB s2r_B)
{
    using namespace cute;

    auto [M, N, K] = shape_MNK;

    // TMA-bound global views (shape from descriptor)
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));   // (M, K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));   // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BM, BK, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BN, BK, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BM, BN)

    // ------------------------------------------------------------------
    // Shared storage: pipelined smem tiles + per-stage mbarriers
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) char smem_raw[];
    cute::half_t* smemA_ptr = reinterpret_cast<cute::half_t*>(smem_raw);

    constexpr int smemA_elems = cosize_v<decltype(sA_layout.layout_b())>;
    constexpr int smemB_elems = cosize_v<decltype(sB_layout.layout_b())>;
    constexpr int smem_half_bytes = (smemA_elems + smemB_elems) * sizeof(cute::half_t);
    constexpr int smem_bar_off    = (smem_half_bytes + 15) & ~15;  // 16B align

    cute::half_t* smemB_ptr = smemA_ptr + smemA_elems;
    uint64_t* full_mbar  = reinterpret_cast<uint64_t*>(smem_raw + smem_bar_off);
    uint64_t* empty_mbar = full_mbar + NUM_STAGES;

    Tensor sA = make_tensor(make_smem_ptr(smemA_ptr), sA_layout); // (BM, BK, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB_ptr), sB_layout); // (BN, BK, PIPE)

    // ------------------------------------------------------------------
    // TMA partitioning (CTA-wide, single CTA ⇒ Int<0>, Layout<_1>)
    // group_modes<0,2> merges (BM,BK) so tma_partition returns (TMA, k) and
    // (TMA, PIPE).
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
    int tid          = threadIdx.x;
    int warp_idx     = tid / 32;
    bool is_producer = (tid < NUM_PRODUCER_THREADS);
    int consumer_tid = tid - NUM_PRODUCER_THREADS;    // 0..NUM_CONSUMER_THREADS-1

    // Elect one thread in WG0 to drive TMA
    bool tma_leader = is_producer && (warp_idx == 0) && cute::elect_one_sync();

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier; // TMA complete
    using EmptyBarrier = cutlass::arch::ClusterBarrier;            // MMA consumed

    // ------------------------------------------------------------------
    // Barrier init (single thread, followed by a CTA-wide fence)
    // ------------------------------------------------------------------
    if (tid == 0) {
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            FullBarrier::init(&full_mbar[s],  1);                         // only tma_leader arrives
            EmptyBarrier::init(&empty_mbar[s], NUM_CONSUMER_THREADS);     // every consumer thread arrives
        }
        cutlass::arch::fence_barrier_init();  // publish mbarrier inits to all async agents
    }
    __syncthreads();

    // ==================================================================
    // PRODUCER (warp group 0, 128 threads, only tma_leader issues TMAs)
    // ==================================================================
    if (is_producer) {
        // Prologue: fill first min(NUM_STAGES, k_tile_count) pipes.
        // empty_mbar is still at its initial phase 0 (consumer hasn't arrived yet),
        // so we MUST NOT advance write_phase here — the steady-state wait below
        // enters at (pipe=0, phase=0), matching the barrier's initial parity.
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            if (s < k_tile_count && tma_leader) {
                FullBarrier::arrive_and_expect_tx(&full_mbar[s], tma_tx_bytes);
                copy(tma_a.with(full_mbar[s]), tAgA(_, s), tAsA(_, s));
                copy(tma_b.with(full_mbar[s]), tBgB(_, s), tBsB(_, s));
            }
        }

        int write_pipe  = 0;
        int write_phase = 0;

        // Mainloop: wait for consumer to release a pipe, then issue next TMA.
        for (int k_tile = NUM_STAGES; k_tile < k_tile_count; ++k_tile) {
            EmptyBarrier::wait(&empty_mbar[write_pipe], write_phase);
            if (tma_leader) {
                FullBarrier::arrive_and_expect_tx(&full_mbar[write_pipe], tma_tx_bytes);
                copy(tma_a.with(full_mbar[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
                copy(tma_b.with(full_mbar[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
            }
            ++write_pipe;
            if (write_pipe == NUM_STAGES) { write_pipe = 0; write_phase ^= 1; }
        }
        return;
    }

    // ==================================================================
    // CONSUMER (warp groups 1+2, 256 threads)
    // ==================================================================

    // MMA & copy partitions use consumer_tid (0..255)
    auto thr_mma = mma.get_slice(consumer_tid);
    Tensor tCgC = thr_mma.partition_C(gC);                          // (MMA, MMA_M, MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                    // (MMA, MMA_M, MMA_N) f32 acc
    clear(tCrC);

    // Frag shape comes from a single pipe slice (layout-only reference).
    Tensor sA_ref = sA(_, _, 0);
    Tensor sB_ref = sB(_, _, 0);
    Tensor tCrA = thr_mma.partition_fragment_A(sA_ref);             // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB_ref);             // (MMA, MMA_N, MMA_K)

    auto s2r_thr_a = s2r_A.get_slice(consumer_tid);
    auto s2r_thr_b = s2r_B.get_slice(consumer_tid);
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);

    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_pipe  = 0;
    int read_phase = 0;

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        // Wait for TMA of this pipe
        FullBarrier::wait(&full_mbar[read_pipe], read_phase);

        // Partition this pipe's smem tile for the s2r copy
        Tensor sA_stg = sA(_, _, read_pipe);
        Tensor sB_stg = sB(_, _, read_pipe);
        Tensor tXsA   = s2r_thr_a.partition_S(sA_stg);
        Tensor tXsB   = s2r_thr_b.partition_S(sB_stg);

        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            copy(s2r_A, tXsA(_, _, k_block), tXrA(_, _, k_block));
            copy(s2r_B, tXsB(_, _, k_block), tXrB(_, _, k_block));
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }

        // Signal producer that this pipe is free
        EmptyBarrier::arrive(&empty_mbar[read_pipe]);

        ++read_pipe;
        if (read_pipe == NUM_STAGES) { read_pipe = 0; read_phase ^= 1; }
    }

    // Epilogue (consumer threads only write C)
    axpby(alpha, tCrC, beta, tCgC);
}


// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 64, int NUM_STAGES = 3>
void hgemm_tma(
    int m, int n, int k,
    float alpha,
    const cute::half_t* A, int ldA,
    const cute::half_t* B, int ldB,
    float beta,
    cute::half_t* C, int ldC)
{
    using namespace cute;

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto shape_MNK = make_shape(m, n, k);

    // TN strides: A(M,K):(ldA,1), B(N,K):(ldB,1), C(M,N):(1,ldC)
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    // ---------------------------------------------------------------
    // Tensor-core MMA: SM80 16x8x16 F32-accum, 256 consumer threads
    // Warp tiling 2×4 (M×N) → 32×32 per issue; BM=BN=128 ⇒ 4×4 iters per pipe.
    // ---------------------------------------------------------------
    auto mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<_2, _4, _1>>{}
    );
    constexpr int NUM_CONSUMER_THREADS = size(decltype(mma){});  // 256
    constexpr int NUM_PRODUCER_THREADS = 128;                    // one warp group
    constexpr int NUM_THREADS          = NUM_CONSUMER_THREADS + NUM_PRODUCER_THREADS; // 384

    // ---------------------------------------------------------------
    // Pipelined smem: GMMA K-major 128-byte swizzle (TMA native + ldmatrix-ok)
    // ---------------------------------------------------------------
    auto sA_layout = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<cute::half_t>{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<NUM_STAGES>{}));
    auto sB_layout = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<cute::half_t>{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<NUM_STAGES>{}));

    // ---------------------------------------------------------------
    // TMA atoms — descriptor built from a single pipe slice
    // ---------------------------------------------------------------
    Tensor mA_desc = make_tensor(A, make_shape(m, k), dA);
    Tensor mB_desc = make_tensor(B, make_shape(n, k), dB);

    auto tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA_desc, sA_layout(_, _, 0),
                              make_shape(Int<BM>{}, Int<BK>{}));
    auto tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB_desc, sB_layout(_, _, 0),
                              make_shape(Int<BN>{}, Int<BK>{}));

    // ---------------------------------------------------------------
    // smem → register: ldmatrix. A uses x4 (16×16 per warp), B uses x2 (16×8).
    // ---------------------------------------------------------------
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, cute::half_t>;
    auto s2r_copy_A = make_tiled_copy_A(S2RCopyAtomA{}, mma);
    auto s2r_copy_B = make_tiled_copy_B(S2RCopyAtomB{}, mma);

    // ---------------------------------------------------------------
    // Dynamic smem sizing: tile buffers + barrier arrays
    // ---------------------------------------------------------------
    constexpr int smemA_elems = cosize_v<decltype(sA_layout.layout_b())>;
    constexpr int smemB_elems = cosize_v<decltype(sB_layout.layout_b())>;
    constexpr int smem_half_bytes = (smemA_elems + smemB_elems) * sizeof(cute::half_t);
    constexpr int smem_bar_off    = (smem_half_bytes + 15) & ~15;
    constexpr int smem_bar_bytes  = 2 * NUM_STAGES * sizeof(uint64_t);
    constexpr int smem_bytes      = smem_bar_off + smem_bar_bytes;

    auto kernel = hgemm_tma_device<NUM_STAGES, NUM_CONSUMER_THREADS, NUM_PRODUCER_THREADS,
        decltype(shape_MNK), decltype(cta_tiler),
        decltype(tmaA), decltype(sA_layout),
        decltype(tmaB), decltype(sB_layout),
        decltype(dC), decltype(mma),
        decltype(s2r_copy_A), decltype(s2r_copy_B)>;

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    dim3 block(NUM_THREADS);
    dim3 grid(size(ceil_div(m, Int<BM>{})),
              size(ceil_div(n, Int<BN>{})));

    kernel<<<grid, block, smem_bytes>>>(
        shape_MNK, cta_tiler, alpha,
        tmaA, sA_layout,
        tmaB, sB_layout,
        beta, C, dC,
        mma, s2r_copy_A, s2r_copy_B);
}
