"""
Simple TMA + WGMMA HGEMM using CuTe DSL (TN): C = alpha * A^T * B + beta * C

Single warp group (128 threads), cooperative mainloop.
All threads participate in both TMA loading and WGMMA compute.
Multi-stage pipeline via PipelineTmaAsync.

Layout convention (TN col-major):
  A stored (K,M) col-major → CuTe tensor (M,K):(K,1) — K-contiguous
  B stored (K,N) col-major → CuTe tensor (N,K):(K,1) — K-contiguous
  C stored (M,N) col-major → CuTe tensor (M,N):(1,M) — in-place output

Matches CUTLASS/hgemm/hgemm_wgmma_tma.cuh.
"""

import sys
import numpy as np
import cupy as cp
import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.cute.runtime import from_dlpack


class HgemmWgmmaTma:
    def __init__(self, bm=128, bn=128, num_stages=3):
        self._bm = bm
        self._bn = bn
        self._bk = None  # set from MMA instruction shape
        self._num_stages = num_stages
        self._tile_shape_mnk = None
        self._tiled_mma = None
        self._mma_warp_groups = 1
        self._threads_per_cta = 128

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,   # (M, K):(K,1)
        mB: cute.Tensor,   # (N, K):(K,1)
        mC: cute.Tensor,   # (M, N):(1,M) — in-place
        alpha: cutlass.Float32 = 1.0,
        beta: cutlass.Float32 = 0.0,
        stream: cuda_driver.CUstream = cuda_driver.CUstream(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        BM, BN = self._bm, self._bn
        S = self._num_stages

        a_layout_enum = utils.LayoutEnum.from_tensor(mA)
        b_layout_enum = utils.LayoutEnum.from_tensor(mB)

        # Create WGMMA tiled MMA
        atom_layout_mnk = (1, 1, 1)
        self._tiled_mma = sm90_utils.make_trivial_tiled_mma(
            mA.element_type, mB.element_type,
            a_layout_enum.sm90_mma_major_mode(),
            b_layout_enum.sm90_mma_major_mode(),
            cutlass.Float32,
            atom_layout_mnk,
            tiler_mn=(64, BN),
        )

        mma_inst_shape_k = cute.size(self._tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self._bk = mma_inst_shape_k * mma_inst_tile_k
        BK = self._bk
        self._tile_shape_mnk = (BM, BN, BK)

        # Swizzled smem layouts
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, self._tile_shape_mnk, mA.element_type, S)
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, self._tile_shape_mnk, mB.element_type, S)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        tma_copy_bytes = (
            cute.size_in_bytes(mA.element_type, a_smem_layout)
            + cute.size_in_bytes(mB.element_type, b_smem_layout)
        )

        # TMA atoms
        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, mA, a_smem_layout, (BM, BK))
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, mB, b_smem_layout, (BN, BK))

        # Shared storage
        buffer_align = 1024

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, S * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(a_smem_layout_staged)],
                buffer_align,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, cute.cosize(b_smem_layout_staged)],
                buffer_align,
            ]

        self._shared_storage = SharedStorage

        grid_dim = (*cute.ceil_div(mC.shape, (BM, BN)), 1)

        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            mC, alpha, beta,
            self._tiled_mma,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ).launch(
            grid=grid_dim,
            block=[self._threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mC: cute.Tensor,
        alpha: cutlass.Float32,
        beta: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        BM, BN, BK = self._tile_shape_mnk
        S = self._num_stages

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        tile_coord = (bidx, bidy, None)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(cutlass.Float16, a_smem_layout)
            + cute.size_in_bytes(cutlass.Float16, b_smem_layout)
        )

        num_warps = self._threads_per_cta // 32
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)

        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=S,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        gA = cute.local_tile(mA_mk, self._tile_shape_mnk, tile_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB_nk, self._tile_shape_mnk, tile_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC,    self._tile_shape_mnk, tile_coord, proj=(1, 1, None))

        # TMA partitions
        sA_for_tma = cute.group_modes(sA, 0, 2)
        gA_for_tma = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sA_for_tma, gA_for_tma)

        sB_for_tma = cute.group_modes(sB, 0, 2)
        gB_for_tma = cute.group_modes(gB, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), sB_for_tma, gB_for_tma)

        # MMA partitions
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC)
        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, cutlass.Float32)

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        k_tile_cnt = cute.size(gA, mode=[2])
        num_k_blocks = cute.size(tCrA, mode=[2])
        k_pipe_mmas = 1

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, S)
        consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, S)
        consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, S)

        prefetch_k_tile_cnt = cutlass.max(cutlass.min(S, k_tile_cnt), 0)

        # Prefetch stages
        if warp_idx == 0:
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(producer_state)
                tAgA_k = tAgA[(None, producer_state.count)]
                tAsA_pipe = tAsA[(None, producer_state.index)]
                tBgB_k = tBgB[(None, producer_state.count)]
                tBsB_pipe = tBsB[(None, producer_state.index)]
                cute.copy(tma_atom_a, tAgA_k, tAsA_pipe,
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                          mcast_mask=0)
                cute.copy(tma_atom_b, tBgB_k, tBsB_pipe,
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                          mcast_mask=0)
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # Prologue MMA
        peek_status = cutlass.Boolean(1)
        if consumer_read_state.count < k_tile_cnt:
            peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, consumer_read_state.index)
                cute.gemm(tiled_mma, accumulators,
                          tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            consumer_read_state.advance()
            peek_status = cutlass.Boolean(1)
            if consumer_read_state.count < k_tile_cnt:
                peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)

        # Mainloop
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, consumer_read_state.index)
                cute.gemm(tiled_mma, accumulators,
                          tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
            mainloop_pipeline.consumer_release(consumer_release_state)
            consumer_read_state.advance()
            consumer_release_state.advance()
            peek_status = cutlass.Boolean(1)
            if consumer_read_state.count < k_tile_cnt:
                peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)
            # Overlap: issue next TMA loads while computing
            if warp_idx == 0 and producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(producer_state)
                tAgA_k = tAgA[(None, producer_state.count)]
                tAsA_pipe = tAsA[(None, producer_state.index)]
                tBgB_k = tBgB[(None, producer_state.count)]
                tBsB_pipe = tBsB[(None, producer_state.index)]
                cute.copy(tma_atom_a, tAgA_k, tAsA_pipe,
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                          mcast_mask=0)
                cute.copy(tma_atom_b, tBgB_k, tBsB_pipe,
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                          mcast_mask=0)
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # Epilogue: drain pipeline and write back
        cute.nvgpu.warpgroup.wait_group(0)
        cute.arch.sync_threads()

        epilogue_f32 = alpha * accumulators.load() + beta * tCgC.load()
        tCrC = cute.make_fragment_like(tCgC)
        tCrC.store(epilogue_f32.to(cutlass.Float16))
        cute.autovec_copy(tCrC, tCgC)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def run_gemm_wgmma_tma(M=512, N=512, K=512, bm=128, bn=128, num_stages=3):
    """Run the WGMMA+TMA HGEMM and verify against NumPy reference."""
    np.random.seed(42)

    # TN col-major: A stored (K,M), B stored (K,N), C stored (M,N)
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))
    C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float16))

    A_d = cp.array(A_h, order="F")
    B_d = cp.array(B_h, order="F")
    C_d = cp.array(C_h, order="F")

    # CuTeDSL expects mA=(M,K):(K,1), mB=(N,K):(K,1), mC=(M,N):(1,M)
    A_t = from_dlpack(A_d.T, assumed_align=16)
    B_t = from_dlpack(B_d.T, assumed_align=16)
    C_t = from_dlpack(C_d, assumed_align=16)

    alpha, beta = 1.0, 0.0
    gemm = HgemmWgmmaTma(bm=bm, bn=bn, num_stages=num_stages)
    gemm(A_t, B_t, C_t, alpha=alpha, beta=beta)
    cp.cuda.Device().synchronize()

    # NumPy reference: D = alpha * A^T @ B + beta * C  (A^T is (M,K), B is (K,N))
    D_ref = alpha * (A_h.T.astype(np.float32) @ B_h.astype(np.float32)) \
            + beta * C_h.astype(np.float32)
    D_out = cp.asnumpy(C_d).astype(np.float32)

    abs_err = float(np.max(np.abs(D_out - D_ref)))
    rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
    passed = bool(np.allclose(D_out, D_ref, atol=5e-2, rtol=5e-2))

    status = "PASS" if passed else "FAIL"
    print(f"hgemm_wgmma_tma ({M}x{N}x{K}): {status}  "
          f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}")
    return passed


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N = int(sys.argv[2]) if len(sys.argv) > 2 else M
    K = int(sys.argv[3]) if len(sys.argv) > 3 else M
    ok = run_gemm_wgmma_tma(M, N, K)
    sys.exit(0 if ok else 1)
