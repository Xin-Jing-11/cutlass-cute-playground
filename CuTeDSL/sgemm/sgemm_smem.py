"""
Smem SGEMM using CuTe DSL: C = alpha * A * B + beta * C

Each thread computes one element. Shared memory tiling with
gmem → smem → register pipeline per K-tile.

A(M,K) col-major, B(N,K) col-major, C(M,N) col-major.
Float32 in/out.
"""

import sys

import numpy as np
import cupy as cp

import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
from cutlass.cute import nvgpu
from cutlass.cute.runtime import from_dlpack


class SgemmSmem:
    """
    Smem SGEMM: BLOCK_SIZE x BLOCK_SIZE threads per CTA.
    Shared memory tiling with smem → register copy before gemm.
    """

    def __init__(self, block_size=32):
        self._block_size = block_size

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M, K) col-major, f32
        mB: cute.Tensor,  # (N, K) col-major, f32
        mC: cute.Tensor,  # (M, N) col-major, f32
        alpha: cutlass.Float32 = 1.0,
        beta: cutlass.Float32 = 0.0,
        stream: cuda_driver.CUstream = cuda_driver.CUstream(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        BS = self._block_size

        # Thread layouts for loading gmem → smem
        tA = cute.make_layout((BS, BS), stride=(1, BS))   # col-major (matches A)
        tB = cute.make_layout((BS, BS), stride=(BS, 1))   # row-major (matches B)
        # Thread layout for compute partitioning
        tC = cute.make_layout((BS, BS), stride=(1, BS))   # col-major

        # Shared memory layouts
        sA_layout = cute.make_layout((BS, BS), stride=(1, BS))   # col-major
        sB_layout = cute.make_layout((BS, BS), stride=(BS, 1))   # row-major

        grid_dim = (*cute.ceil_div(mC.shape, (BS, BS)), 1)

        self.kernel(mA, mB, mC, alpha, beta, tA, tB, tC, sA_layout, sB_layout).launch(
            grid=grid_dim,
            block=[BS * BS, 1, 1],
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
        tA: cute.Layout,
        tB: cute.Layout,
        tC: cute.Layout,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
    ):
        BS = self._block_size
        cta_tiler = (BS, BS, BS)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta_coord = (bidx, bidy, None)

        # Global tiles
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=cta_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=cta_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=cta_coord, proj=(1, 1, None))

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float32, sA_layout)
        sB = smem.allocate_tensor(cutlass.Float32, sB_layout)

        # Thread partitions for loading gmem → smem
        tAgA = cute.local_partition(gA, tA, tidx)   # (1, 1, k)
        tAsA = cute.local_partition(sA, tA, tidx)   # (1, 1)
        tBgB = cute.local_partition(gB, tB, tidx)   # (1, 1, k)
        tBsB = cute.local_partition(sB, tB, tidx)   # (1, 1)

        # Thread partitions for compute
        tCgC = cute.local_partition(gC, tC, tidx)                  # (1, 1)
        tCsA = cute.local_partition(sA, tC, tidx, proj=(1, None))  # (1, BLK_K)
        tCsB = cute.local_partition(sB, tC, tidx, proj=(None, 1))  # (1, BLK_K)

        # Register fragments (smem → register before gemm)
        tCrA = cute.make_fragment_like(tCsA)  # (1, BLK_K) in registers
        tCrB = cute.make_fragment_like(tCsB)  # (1, BLK_K) in registers
        tCrC = cute.make_fragment(tCgC.shape, cutlass.Float32)
        tCrC.fill(0.0)

        # MMA atom: scalar UniversalFMA
        mma_atom = cute.make_mma_atom(nvgpu.MmaUniversalOp(cutlass.Float32))

        # Mainloop: load gmem → smem → registers, then gemm
        k_max = cute.size(tAgA, mode=[2])
        for k_tile in cutlass.range(k_max):
            # gmem → smem
            cute.basic_copy(tAgA[None, None, k_tile], tAsA)
            cute.basic_copy(tBgB[None, None, k_tile], tBsB)
            cute.arch.sync_threads()
            # smem → register
            cute.autovec_copy(tCsA, tCrA)
            cute.autovec_copy(tCsB, tCrB)
            # register × register → register
            cute.gemm(mma_atom, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        # Epilogue in-place: C = alpha * A*B + beta * C
        epilogue_op = lambda acc: alpha * acc + beta * tCgC.load()
        tCgC.store(epilogue_op(tCrC.load()))
