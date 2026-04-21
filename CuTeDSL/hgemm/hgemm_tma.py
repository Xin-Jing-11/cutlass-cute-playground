"""
TMA-loaded tensor-core HGEMM using CuTe DSL (TN):
  C = alpha * A^T * B + beta * C

**Status: WIP.** Compiles cleanly, but launches hang in the mbarrier-wait
spin on SM120 (the barrier never flips — TMA complete_tx / expect_tx
book-keeping on SM120 via the DSL is subtle and not yet working here).
Kept as a starting point; NOT wired into CuTeDSL/hgemm/instantiate.py.

Known-good alternative: use pipeline.PipelineTmaAsync (see hopper/dense_gemm.py
in the CUTLASS DSL examples) — its create() needs a cluster layout and a
SharedStorage dataclass, which is heavier than this playground's pattern.

A stored (K,M) col-major → CuTe (M,K):(K,1).
B stored (K,N) col-major → CuTe (N,K):(K,1).
C stored (M,N) col-major → CuTe (M,N):(1,M)  — in-place.
Half in/out, float32 accumulator.
"""

import sys

import numpy as np
import cupy as cp

import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warp import MmaF16BF16Op, LdMatrix8x8x16bOp
from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileG2SOp, make_tiled_tma_atom, tma_partition
from cutlass.cute.runtime import from_dlpack


class HgemmTma:
    """Single-buffered TMA HGEMM.

    - 4 warps (128 threads), 2×2×1 SM80_16x8x16 MMA warp tiling.
    - One mbarrier (count=NUM_THREADS, TMA drops tx_bytes onto it).
    - Every tile: thread 0 issues cp.async.bulk.tensor; all threads arrive;
      all threads wait; MMA; re-init barrier; next tile.
    """

    def __init__(self, bm=128, bn=128, bk=64):
        self._bm = bm
        self._bn = bn
        self._bk = bk
        assert bk % 16 == 0, "BK must be divisible by MMA_K=16"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M, K):(K,1)
        mB: cute.Tensor,  # (N, K):(K,1)
        mC: cute.Tensor,  # (M, N):(1,M)  — in-place
        alpha: cutlass.Float32 = 1.0,
        beta: cutlass.Float32 = 0.0,
        stream: cuda_driver.CUstream = cuda_driver.CUstream(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        BM, BN, BK = self._bm, self._bn, self._bk

        mma_op = MmaF16BF16Op(
            ab_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16))
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((2, 2, 1)))
        num_threads = 128

        # Plain (no swizzle) smem layouts, row-major (M,K) and (N,K).
        sA_layout = cute.make_layout((BM, BK), stride=(BK, 1))
        sB_layout = cute.make_layout((BN, BK), stride=(BK, 1))

        tma_atom_A, gA_tma = make_tiled_tma_atom(
            CopyBulkTensorTileG2SOp(), mA, sA_layout, (BM, BK))
        tma_atom_B, gB_tma = make_tiled_tma_atom(
            CopyBulkTensorTileG2SOp(), mB, sB_layout, (BN, BK))

        s2r_atom_A = cute.make_copy_atom(
            LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16)
        s2r_atom_B = cute.make_copy_atom(
            LdMatrix8x8x16bOp(transpose=False, num_matrices=2), cutlass.Float16)
        s2r_copy_A = cute.make_tiled_copy_A(s2r_atom_A, tiled_mma)
        s2r_copy_B = cute.make_tiled_copy_B(s2r_atom_B, tiled_mma)

        tx_bytes = (BM * BK + BN * BK) * 2  # 2 B / half

        grid_dim = (*cute.ceil_div(mC.shape, (BM, BN)), 1)

        self.kernel(
            tma_atom_A, gA_tma,
            tma_atom_B, gB_tma,
            mC, alpha, beta,
            tiled_mma, s2r_copy_A, s2r_copy_B,
            sA_layout, sB_layout,
            tx_bytes,
        ).launch(
            grid=grid_dim,
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_A: cute.CopyAtom,
        gA_tma: cute.Tensor,
        tma_atom_B: cute.CopyAtom,
        gB_tma: cute.Tensor,
        mC: cute.Tensor,
        alpha: cutlass.Float32,
        beta: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        s2r_A: cute.TiledCopy,
        s2r_B: cute.TiledCopy,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tx_bytes: cutlass.Constexpr,
    ):
        BM, BN, BK = self._bm, self._bn, self._bk
        NUM_THREADS = 128
        cta_tiler = (BM, BN, BK)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta_coord = (bidx, bidy, None)

        gC = cute.local_tile(mC, tiler=cta_tiler, coord=cta_coord, proj=(1, 1, None))
        gA = cute.local_tile(gA_tma, tiler=cta_tiler, coord=cta_coord, proj=(1, None, 1))
        gB = cute.local_tile(gB_tma, tiler=cta_tiler, coord=cta_coord, proj=(None, 1, 1))

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float16, sA_layout, byte_alignment=128)
        sB = smem.allocate_tensor(cutlass.Float16, sB_layout, byte_alignment=128)
        mbar = smem.allocate_array(cutlass.Int64, num_elems=1)

        # Single-stage TMA partition: (TMA_chunk, k)
        tAsA, tAgA = tma_partition(
            tma_atom_A, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = tma_partition(
            tma_atom_B, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        # MMA partitions
        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        thr_s2r_a = s2r_A.get_slice(tidx)
        tXrA = thr_s2r_a.retile(tCrA)
        tXsA = thr_s2r_a.partition_S(sA)
        thr_s2r_b = s2r_B.get_slice(tidx)
        tXrB = thr_s2r_b.retile(tCrB)
        tXsB = thr_s2r_b.partition_S(sB)

        k_max = cute.size(tAgA, mode=[1])
        k_block_max = cute.size(tCrA, mode=[2])

        # Iterate one K-tile at a time; re-init barrier each iteration so we
        # can keep using phase 0 every time (simplest correct pattern).
        for k_tile in cutlass.range(k_max):
            # Init barrier (thread 0) + publish + sync
            if tidx == 0:
                cute.arch.mbarrier_init(mbar, NUM_THREADS)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # Leader thread issues TMA + arrive_and_expect_tx(1 + tx_bytes).
            # Other 127 threads just arrive (count=NUM_THREADS ensures all participate).
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, tx_bytes)
                cute.copy(tma_atom_A, tAgA[None, k_tile], tAsA, tma_bar_ptr=mbar)
                cute.copy(tma_atom_B, tBgB[None, k_tile], tBsB, tma_bar_ptr=mbar)
            else:
                cute.arch.mbarrier_arrive(mbar)

            cute.arch.mbarrier_wait(mbar, 0)

            # MMA
            for k_block in cutlass.range(k_block_max, unroll_full=True):
                cute.copy(s2r_A, tXsA[None, None, k_block], tXrA[None, None, k_block])
                cute.copy(s2r_B, tXsB[None, None, k_block], tXrB[None, None, k_block])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                          tCrB[None, None, k_block], tCrC)

            # Ensure everyone finished reading sA/sB before re-init next iter.
            cute.arch.sync_threads()

        # Epilogue
        epilogue_f32 = alpha * tCrC.load() + beta * tCgC.load()
        tCrC.store(epilogue_f32)
        tCrD = cute.make_fragment_like(tCgC)
        tCrD.store(tCrC.load().to(cutlass.Float16))
        cute.autovec_copy(tCrD, tCgC)


def run_gemm_tma(M, N, K, bm=128, bn=128, bk=64):
    np.random.seed(42)
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))

    A_d = cp.array(A_h, order='F')
    B_d = cp.array(B_h, order='F')
    C_d = cp.zeros((M, N), dtype=cp.float16, order='F')

    A_t = from_dlpack(A_d.T, assumed_align=16)
    B_t = from_dlpack(B_d.T, assumed_align=16)
    C_t = from_dlpack(C_d, assumed_align=16)

    gemm = HgemmTma(bm=bm, bn=bn, bk=bk)
    gemm(A_t, B_t, C_t, alpha=1.0, beta=0.0)

    C_out = cp.asnumpy(C_d).astype(np.float32)
    D_ref = A_h.T.astype(np.float32) @ B_h.astype(np.float32)
    abs_err = np.max(np.abs(C_out - D_ref))
    ref_norm = np.max(np.abs(D_ref)) + 1e-6
    rel_err = abs_err / ref_norm
    print(f"M={M} N={N} K={K} BM={bm} BN={bn} BK={bk}  "
          f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}  "
          f"{'PASS' if rel_err < 0.05 else 'FAIL'}")
    return C_out


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    run_gemm_tma(M, N, K)
