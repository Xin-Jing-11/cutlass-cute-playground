"""
Tensor-core HGEMM using CuTe DSL (TN): D = alpha * A^T * B + beta * C

Mirrors CUTLASS/hgemm/hgemm_mma.cuh:
  - SM80 warp-level MMA: 16×8×16, F16 inputs, F32 accumulator
  - Tiled 2×2×1 warps → 128 threads, 32×16 per MMA step
  - 128-bit vectorized gmem→smem copies
  - Scalar smem→register copies retiled to match MMA layout
  - No swizzle: CuTeDSL has a known bug where ComposedLayout + multi-K
    s2r iteration produces wrong results (works in C++ CuTe).
    The C++ version uses Swizzle<2,3,6> on both sA and sB.

A stored (K,M) col-major → CuTe (M,K):(K,1).
B stored (K,N) col-major → CuTe (N,K):(K,1).
C stored (M,N) col-major → CuTe (M,N):(1,M)  — in-place: C = alpha * A^T * B + beta * C.
Half in/out, float32 accumulator.
"""

import sys

import numpy as np
import cupy as cp

import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.cute.runtime import from_dlpack


class HgemmMma:
    """
    Tensor-core HGEMM with swizzled shared memory and 128-bit vectorized copies.
    Uses SM80_16x8x16_F32F16F16F32_TN MMA with 2×2×1 warp tiling (128 threads).
    """

    def __init__(self, bm=128, bn=128, bk=32):
        self._bm = bm
        self._bn = bn
        self._bk = bk
        assert bk % 16 == 0, "BK must be divisible by MMA_K=16"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M, K):(K,1) K-contiguous, f16
        mB: cute.Tensor,  # (N, K):(K,1) K-contiguous, f16
        mC: cute.Tensor,  # (M, N):(1,M) col-major, f16  — in-place output
        alpha: cutlass.Float32 = 1.0,
        beta: cutlass.Float32 = 0.0,
        stream: cuda_driver.CUstream = cuda_driver.CUstream(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        BM, BN, BK = self._bm, self._bn, self._bk
        VEC = 8                         # 8 × half = 128 bits
        BK_VEC = BK // VEC             # 4

        # ---------------------------------------------------------------
        # Tensor-core MMA: 16×8×16 with F32 accumulator
        # Tile 2 warps in M, 2 in N → 4 warps = 128 threads
        # Per-step coverage: 32 in M, 16 in N
        # ---------------------------------------------------------------
        mma_op = MmaF16BF16Op(
            ab_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((2, 2, 1)))     # (warpM, warpN, warpK)

        num_threads = 128  # 4 warps × 32 threads

        # ---------------------------------------------------------------
        # Shared memory: LayoutRight (K-contiguous), no swizzle
        #
        # NOTE: The C++ version uses Swizzle<2,3,6> on both sA (atom 16×BK)
        # and sB (atom 8×BK). CuTeDSL has a known bug where ComposedLayout
        # with multi-K s2r iteration (k_block_max > 1) produces wrong results.
        # With MMA_K=16, BK=32 → k_block_max=2, triggering the bug.
        # ---------------------------------------------------------------
        sA_layout = cute.make_layout((BM, BK), stride=(BK, 1))
        sB_layout = cute.make_layout((BN, BK), stride=(BK, 1))

        # ---------------------------------------------------------------
        # gmem → smem: 128-bit vectorized copies
        # Thread layout (ThrM, BK_VEC) LayoutRight, Value layout (1, VEC)
        # ---------------------------------------------------------------
        vec_copy_atom = cute.make_copy_atom(
            nvgpu.CopyUniversalOp(), mA.element_type,
            num_bits_per_copy=128)

        thr_m = num_threads // BK_VEC
        g2s_thr_layout = cute.make_layout((thr_m, BK_VEC), stride=(BK_VEC, 1))
        g2s_val_layout = cute.make_layout((1, VEC))

        tiled_copy_A = cute.make_tiled_copy_tv(
            vec_copy_atom, g2s_thr_layout, g2s_val_layout)
        tiled_copy_B = cute.make_tiled_copy_tv(
            vec_copy_atom, g2s_thr_layout, g2s_val_layout)

        # ---------------------------------------------------------------
        # smem → register: scalar copies, retiled to match MMA layout
        # ---------------------------------------------------------------
        s2r_atom = cute.make_copy_atom(
            nvgpu.CopyUniversalOp(), cutlass.Float16,
            num_bits_per_copy=cutlass.Float16.width)
        s2r_copy_A = cute.make_tiled_copy_A(s2r_atom, tiled_mma)
        s2r_copy_B = cute.make_tiled_copy_B(s2r_atom, tiled_mma)

        # ---------------------------------------------------------------
        # Launch
        # ---------------------------------------------------------------
        grid_dim = (*cute.ceil_div(mC.shape, (BM, BN)), 1)

        self.kernel(
            mA, mB, mC, alpha, beta,
            tiled_copy_A, tiled_copy_B,
            tiled_mma, s2r_copy_A, s2r_copy_B,
            sA_layout, sB_layout,
        ).launch(
            grid=grid_dim,
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        alpha: cutlass.Float32,
        beta: cutlass.Float32,
        g2s_A: cute.TiledCopy,
        g2s_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        s2r_A: cute.TiledCopy,
        s2r_B: cute.TiledCopy,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
    ):
        BM, BN, BK = self._bm, self._bn, self._bk
        cta_tiler = (BM, BN, BK)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta_coord = (bidx, bidy, None)

        # Global tiles
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=cta_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=cta_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=cta_coord, proj=(1, 1, None))

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float16, sA_layout)
        sB = smem.allocate_tensor(cutlass.Float16, sB_layout)

        # --- gmem → smem partitions ---
        thr_g2s_a = g2s_A.get_slice(tidx)
        tAgA = thr_g2s_a.partition_S(gA)    # (CPY, CPY_M, CPY_K, k)
        tAsA = thr_g2s_a.partition_D(sA)    # (CPY, CPY_M, CPY_K)

        thr_g2s_b = g2s_B.get_slice(tidx)
        tBgB = thr_g2s_b.partition_S(gB)
        tBsB = thr_g2s_b.partition_D(sB)

        # --- MMA partitions ---
        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC)                              # (MMA, MMA_M, MMA_N)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))   # (MMA, MMA_M, MMA_K)
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))   # (MMA, MMA_N, MMA_K)
        tCrC = tiled_mma.make_fragment_C(tCgC)                      # (MMA, MMA_M, MMA_N)
        tCrC.fill(0.0)

        # --- smem → register partitions ---
        thr_s2r_a = s2r_A.get_slice(tidx)
        tXsA = thr_s2r_a.partition_S(sA)    # (CPY, MMA_M, MMA_K)
        tXrA = thr_s2r_a.retile(tCrA)       # (CPY, MMA_M, MMA_K)

        thr_s2r_b = s2r_B.get_slice(tidx)
        tXsB = thr_s2r_b.partition_S(sB)    # (CPY, MMA_N, MMA_K)
        tXrB = thr_s2r_b.retile(tCrB)       # (CPY, MMA_N, MMA_K)

        # Mainloop
        k_max = cute.size(tAgA, mode=[3])       # K tiles
        k_block_max = cute.size(tCrA, mode=[2])  # inner MMA-K iterations per tile
        for k_tile in cutlass.range(k_max):
            # gmem → smem (128-bit vectorized)
            cute.copy(g2s_A, tAgA[None, None, None, k_tile], tAsA)
            cute.copy(g2s_B, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.sync_threads()

            # smem → register, then MMA
            for k_block in cutlass.range(k_block_max, unroll_full=True):
                cute.copy(s2r_A, tXsA[None, None, k_block], tXrA[None, None, k_block])
                cute.copy(s2r_B, tXsB[None, None, k_block], tXrB[None, None, k_block])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                          tCrB[None, None, k_block], tCrC)
            cute.arch.sync_threads()

        # Epilogue: C = alpha * acc + beta * C  (f32 compute → f16 in-place)
        epilogue_f32 = alpha * tCrC.load() + beta * tCgC.load()
        tCrC.store(epilogue_f32)
        tCrD = cute.make_fragment_like(tCgC)
        tCrD.store(tCrC.load().to(cutlass.Float16))
        cute.autovec_copy(tCrD, tCgC)


def run_gemm_mma(M, N, K):
    """Run HGEMM MMA and verify against numpy (f32 reference)."""
    np.random.seed(42)
    # TN: A stored (K,M) col-major, B stored (K,N) col-major
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))

    A_d = cp.array(A_h, order='F')
    B_d = cp.array(B_h, order='F')
    C_d = cp.zeros((M, N), dtype=cp.float16, order='F')

    # CuTeDSL sees transposed views: A_t=(M,K):(K,1), B_t=(N,K):(K,1)
    A_t = from_dlpack(A_d.T, assumed_align=16)
    B_t = from_dlpack(B_d.T, assumed_align=16)
    C_t = from_dlpack(C_d, assumed_align=16)

    gemm = HgemmMma()
    gemm(A_t, B_t, C_t, alpha=1.0, beta=0.0)

    C_out = cp.asnumpy(C_d).astype(np.float32)
    D_ref = A_h.T.astype(np.float32) @ B_h.astype(np.float32)
    abs_err = np.max(np.abs(C_out - D_ref))
    ref_norm = np.max(np.abs(D_ref)) + 1e-6
    rel_err = abs_err / ref_norm
    print(f"M={M} N={N} K={K}  abs_err={abs_err:.3e}  rel_err={rel_err:.3e}  "
          f"{'PASS' if rel_err < 0.05 else 'FAIL'}")
    return C_out


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    run_gemm_mma(M, N, K)
