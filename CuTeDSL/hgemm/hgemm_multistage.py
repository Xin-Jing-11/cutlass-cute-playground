"""
Multistage tensor-core HGEMM using CuTe DSL (TN): C = alpha * A^T * B + beta * C

Mirrors CUTLASS/hgemm/hgemm_multistage.cuh:
  - SM80 16x8x16 MMA, 2x2x1 warp tiling (128 threads)
  - cp.async pipelined gmem→smem with NUM_STAGES smem buffers
  - Circular pipeline: prologue fills S-1 stages, mainloop waits for oldest
    ready stage, computes on it, issues next load.

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
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu.warp import MmaF16BF16Op, LdMatrix8x8x16bOp
from cutlass.cute.nvgpu.cpasync import CopyG2SOp, LoadCacheMode
from cutlass.cute.runtime import from_dlpack


class HgemmMultistage:
    """
    Multistage HGEMM: NUM_STAGES cp.async-pipelined smem buffers.
    2×2×1 warp tiling (128 threads) with SM80 tensor-core MMA.
    NUM_STAGES=2 is double buffering.
    """

    def __init__(self, bm=128, bn=128, bk=32, num_stages=3):
        self._bm = bm
        self._bn = bn
        self._bk = bk
        self._num_stages = num_stages
        assert bk % 16 == 0, "BK must be divisible by MMA_K=16"
        assert num_stages >= 2, "NUM_STAGES must be >= 2"

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
        S = self._num_stages
        VEC = 8                     # 8 × half = 128 bits
        BK_VEC = BK // VEC

        # Tensor-core MMA, same as HgemmMma
        mma_op = MmaF16BF16Op(
            ab_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((2, 2, 1)))

        num_threads = 128

        # Multistage smem layout: (BM, BK, S) with stage as outer mode.
        # Stride (BK, 1, BM*BK) packs each stage contiguously.
        sA_layout = cute.make_layout((BM, BK, S), stride=(BK, 1, BM * BK))
        sB_layout = cute.make_layout((BN, BK, S), stride=(BK, 1, BN * BK))

        # gmem → smem: cp.async 128-bit copies (bypass L1)
        vec_copy_atom = cute.make_copy_atom(
            CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL),
            mA.element_type,
            num_bits_per_copy=128)

        thr_m = num_threads // BK_VEC
        g2s_thr_layout = cute.make_layout((thr_m, BK_VEC), stride=(BK_VEC, 1))
        g2s_val_layout = cute.make_layout((1, VEC))

        tiled_copy_A = cute.make_tiled_copy_tv(
            vec_copy_atom, g2s_thr_layout, g2s_val_layout)
        tiled_copy_B = cute.make_tiled_copy_tv(
            vec_copy_atom, g2s_thr_layout, g2s_val_layout)

        # smem → register: ldmatrix (warp-cooperative 8x8 half tile loads)
        s2r_atom_A = cute.make_copy_atom(
            LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float16)
        s2r_atom_B = cute.make_copy_atom(
            LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
            cutlass.Float16)
        s2r_copy_A = cute.make_tiled_copy_A(s2r_atom_A, tiled_mma)
        s2r_copy_B = cute.make_tiled_copy_B(s2r_atom_B, tiled_mma)

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
        S = self._num_stages
        cta_tiler = (BM, BN, BK)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta_coord = (bidx, bidy, None)

        gA = cute.local_tile(mA, tiler=cta_tiler, coord=cta_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=cta_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=cta_coord, proj=(1, 1, None))

        # Allocate multistage smem
        smem = cutlass.utils.SmemAllocator()
        sA_all = smem.allocate_tensor(cutlass.Float16, sA_layout)   # (BM, BK, S)
        sB_all = smem.allocate_tensor(cutlass.Float16, sB_layout)   # (BN, BK, S)

        # --- gmem → smem partitions (full, with stage mode) ---
        thr_g2s_a = g2s_A.get_slice(tidx)
        tAgA = thr_g2s_a.partition_S(gA)                   # (CPY, CPY_M, CPY_K, k)
        tAsA_all = thr_g2s_a.partition_D(sA_all)           # (CPY, CPY_M, CPY_K, S)

        thr_g2s_b = g2s_B.get_slice(tidx)
        tBgB = thr_g2s_b.partition_S(gB)
        tBsB_all = thr_g2s_b.partition_D(sB_all)

        # --- MMA partitions using stage-0 as shape reference ---
        sA_ref = sA_all[None, None, 0]
        sB_ref = sB_all[None, None, 0]

        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_ref))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_ref))
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        # --- smem → register retile (shape-only) ---
        thr_s2r_a = s2r_A.get_slice(tidx)
        tXrA = thr_s2r_a.retile(tCrA)

        thr_s2r_b = s2r_B.get_slice(tidx)
        tXrB = thr_s2r_b.retile(tCrB)

        k_max = cute.size(tAgA, mode=[3])
        k_block_max = cute.size(tCrA, mode=[2])

        # ========== Prologue: issue stages 0 .. S-2 ==========
        for s in cutlass.range_constexpr(S - 1):
            cute.copy(
                g2s_A,
                tAgA[None, None, None, s],
                tAsA_all[None, None, None, s])
            cute.copy(
                g2s_B,
                tBgB[None, None, None, s],
                tBsB_all[None, None, None, s])
            cute.arch.cp_async_commit_group()

        # ========== Mainloop ==========
        for k_tile in cutlass.range(k_max):
            # Wait: allow at most S-2 groups in flight → oldest is done
            cute.arch.cp_async_wait_group(S - 2)
            cute.arch.sync_threads()

            # Compute on ready stage = k_tile % S
            read_stage = k_tile % S
            sA_r = sA_all[None, None, read_stage]
            sB_r = sB_all[None, None, read_stage]
            tXsA = thr_s2r_a.partition_S(sA_r)
            tXsB = thr_s2r_b.partition_S(sB_r)

            for k_block in cutlass.range(k_block_max, unroll_full=True):
                cute.copy(s2r_A, tXsA[None, None, k_block], tXrA[None, None, k_block])
                cute.copy(s2r_B, tXsB[None, None, k_block], tXrB[None, None, k_block])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                          tCrB[None, None, k_block], tCrC)

            # Issue next load (if remaining tiles)
            next_tile = k_tile + (S - 1)
            if next_tile < k_max:
                write_stage = next_tile % S
                cute.copy(
                    g2s_A,
                    tAgA[None, None, None, next_tile],
                    tAsA_all[None, None, None, write_stage])
                cute.copy(
                    g2s_B,
                    tBgB[None, None, None, next_tile],
                    tBsB_all[None, None, None, write_stage])
            cute.arch.cp_async_commit_group()

            cute.arch.sync_threads()

        # Epilogue: C = alpha * acc + beta * C  (f32 → f16 in-place)
        epilogue_f32 = alpha * tCrC.load() + beta * tCgC.load()
        tCrC.store(epilogue_f32)
        tCrD = cute.make_fragment_like(tCgC)
        tCrD.store(tCrC.load().to(cutlass.Float16))
        cute.autovec_copy(tCrD, tCgC)


def run_gemm_multistage(M, N, K, num_stages=3):
    np.random.seed(42)
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))

    A_d = cp.array(A_h, order='F')
    B_d = cp.array(B_h, order='F')
    C_d = cp.zeros((M, N), dtype=cp.float16, order='F')

    A_t = from_dlpack(A_d.T, assumed_align=16)
    B_t = from_dlpack(B_d.T, assumed_align=16)
    C_t = from_dlpack(C_d, assumed_align=16)

    gemm = HgemmMultistage(num_stages=num_stages)
    gemm(A_t, B_t, C_t, alpha=1.0, beta=0.0)

    C_out = cp.asnumpy(C_d).astype(np.float32)
    D_ref = A_h.T.astype(np.float32) @ B_h.astype(np.float32)
    abs_err = np.max(np.abs(C_out - D_ref))
    ref_norm = np.max(np.abs(D_ref)) + 1e-6
    rel_err = abs_err / ref_norm
    print(f"M={M} N={N} K={K} stages={num_stages}  "
          f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}  "
          f"{'PASS' if rel_err < 0.05 else 'FAIL'}")
    return C_out


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    S = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    run_gemm_multistage(M, N, K, num_stages=S)
