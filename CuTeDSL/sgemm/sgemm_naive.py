"""
Naive SGEMM using CuTe DSL: C = alpha * A * B + beta * C

Each thread computes one element. No shared memory, no tiling.
Global memory loads directly into registers, scalar FMA loop.

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


class SgemmNaive:
    """
    Naive SGEMM: BLOCK_SIZE x BLOCK_SIZE threads per CTA, K=1 tile.
    No shared memory, no register tiling.
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
        tC = cute.make_layout((BS, BS), stride=(1, BS))

        grid_dim = (*cute.ceil_div(mC.shape, (BS, BS)), 1)

        self.kernel(mA, mB, mC, alpha, beta, tC).launch(
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
        tC: cute.Layout,
    ):
        BS = self._block_size
        cta_tiler = (BS, BS, 1)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta_coord = (bidx, bidy, None)

        gA = cute.local_tile(mA, tiler=cta_tiler, coord=cta_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=cta_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=cta_coord, proj=(1, 1, None))

        tCgA = cute.local_partition(gA, tC, tidx, proj=(1, None))
        tCgB = cute.local_partition(gB, tC, tidx, proj=(None, 1))
        tCgC = cute.local_partition(gC, tC, tidx)

        tCrA = cute.make_fragment_like(tCgA[None, None, 0])
        tCrB = cute.make_fragment_like(tCgB[None, None, 0])
        tCrC = cute.make_fragment(tCgC.shape, cutlass.Float32)
        tCrC.fill(0.0)

        # MMA atom: scalar UniversalFMA
        mma_atom = cute.make_mma_atom(nvgpu.MmaUniversalOp(cutlass.Float32))

        # Mainloop: tCrC += tCrA * tCrB via gemm
        k_max = cute.size(tCgA, mode=[2])
        for k_tile in cutlass.range(k_max):
            cute.autovec_copy(tCgA[None, None, k_tile], tCrA)
            cute.autovec_copy(tCgB[None, None, k_tile], tCrB)
            cute.gemm(mma_atom, tCrC, tCrA, tCrB, tCrC)

        # Epilogue in-place: C = alpha * A*B + beta * C
        epilogue_op = lambda acc: alpha * acc + beta * tCgC.load()
        tCgC.store(epilogue_op(tCrC.load()))
