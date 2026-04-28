"""
Persistent warp-specialized WGMMA+TMA HGEMM with (2,1) cluster multicast (TN):
    C = alpha * A^T * B + beta * C

Extends the persistent kernel with a (2,1) cluster: 2 CTAs along M share the
same B tile via TMA multicast.

Layout convention (TN col-major):
  A stored (K,M) col-major -> CuTe tensor (M,K):(K,1) -- K-contiguous
  B stored (K,N) col-major -> CuTe tensor (N,K):(K,1) -- K-contiguous
  C stored (M,N) col-major -> CuTe tensor (M,N):(1,M) -- in-place output
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
from cutlass.cute.runtime import from_dlpack


class HgemmCluster:
    def __init__(self, bm=128, bn=256, num_consumer_warpgroups=2, num_stages=4,
                 cluster_m=2, cluster_n=1):
        self._bm = bm
        self._bn = bn
        self._bk = None
        self._num_stages = num_stages
        self._num_consumer_warpgroups = num_consumer_warpgroups
        self._num_dma_warp_groups = 1
        self._threads_per_cta = (1 + num_consumer_warpgroups) * 128
        self._load_register_requirement = 40
        self._mma_register_requirement = 232
        self._tile_shape_mnk = None
        self._tiled_mma = None
        self._cluster_m = cluster_m
        self._cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        alpha: cutlass.Float32 = 1.0,
        beta: cutlass.Float32 = 0.0,
        stream: cuda_driver.CUstream = cuda_driver.CUstream(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        BM, BN = self._bm, self._bn
        S = self._num_stages
        cluster_m = self._cluster_m
        cluster_n = self._cluster_n

        a_layout_enum = utils.LayoutEnum.from_tensor(mA)
        b_layout_enum = utils.LayoutEnum.from_tensor(mB)

        atom_layout_mnk = (self._num_consumer_warpgroups, 1, 1)
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

        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, self._tile_shape_mnk, mA.element_type, S)
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, self._tile_shape_mnk, mB.element_type, S)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        # TMA atoms: A uses regular G2S, B uses multicast G2S
        tma_op_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op_a, mA, a_smem_layout, (BM, BK),
            num_multicast=1,
        )

        tma_op_b = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op_b, mB, b_smem_layout, (BN, BK),
            num_multicast=cluster_m,
        )

        # CTA layout for the (2,1,1) cluster
        cta_layout_mnk = cute.make_layout((cluster_m, cluster_n, 1))

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

        # Persistent tile scheduling
        c_shape = cute.slice_(self._tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(mC, tiler=c_shape)
        num_ctas_mn = gc[(0, (None, None))].shape
        num_ctas_mnl = (*num_ctas_mn, 1)
        cluster_shape_mnl = (cluster_m, cluster_n, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=1,
            raster_along_m=True,
        )
        max_active_clusters = 132
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            mC, alpha, beta,
            self._tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self._threads_per_cta, 1, 1],
            cluster=(cluster_m, cluster_n, 1),
            min_blocks_per_mp=1,
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
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        BM, BN, BK = self._tile_shape_mnk
        S = self._num_stages
        cluster_m = self._cluster_m
        cluster_n = self._cluster_n

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # Cluster coordination
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # Multicast masks: B is multicast along M dim (mode=0), A is not multicast
        a_mcast_mask = 0
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(cutlass.Float16, a_smem_layout)
            + cute.size_in_bytes(cutlass.Float16, b_smem_layout)
        )

        # Pipeline: producer is 1 DMA thread, consumer arrive count includes mcast
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = cluster_m + cluster_n - 1
        consumer_arrive_cnt = mcast_size * self._num_consumer_warpgroups * 4
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=S,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        pipeline_init_arrive(
            cluster_shape_mn=(cluster_m, cluster_n), is_relaxed=True
        )

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # Global tiles with ALL tile indices as None for persistent scheduling
        gA_mkl = cute.local_tile(
            mA_mk,
            cute.slice_(self._tile_shape_mnk, (None, 0, None)),
            (None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nk,
            cute.slice_(self._tile_shape_mnk, (0, None, None)),
            (None, None),
        )
        gC_mnl = cute.local_tile(
            mC,
            cute.slice_(self._tile_shape_mnk, (None, None, 0)),
            (None, None),
        )

        # TMA partitions with cluster-aware CTA coordinates
        # A: partitioned along N dim of cluster (cluster_n=1)
        a_cta_crd = cluster_coord_mnk[1]
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # B: partitioned along M dim of cluster (cluster_m=2)
        b_cta_crd = cluster_coord_mnk[0]
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # MMA partitions (warp-group level for WGMMA A/B)
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        mma_wg_thread_layout = cute.make_layout(
            self._num_consumer_warpgroups, stride=128
        )
        thr_mma = tiled_mma.get_slice(
            mma_wg_thread_layout(warp_group_idx - self._num_dma_warp_groups)
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # Per-thread C partition for epilogue register-to-gmem copy
        mma_thread_idx = tidx - self._num_dma_warp_groups * 128
        thr_mma_c = tiled_mma.get_slice(mma_thread_idx)
        tCgC = thr_mma_c.partition_C(gC_mnl)

        gC_first_tile = gC_mnl[(None, None, 0, 0)]
        tCgC_first = thr_mma_c.partition_C(gC_first_tile)
        acc_shape = tCgC_first.shape
        accumulators = cute.make_rmem_tensor(acc_shape, cutlass.Float32)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        pipeline_init_wait(cluster_shape_mn=(cluster_m, cluster_n))

        is_dma_warp_group = warp_group_idx < self._num_dma_warp_groups

        # ---- PRODUCER (DMA warp group) ----
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self._load_register_requirement)

        if warp_idx == 0:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, S
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None)]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None)]

                producer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(producer_state)

                    cute.copy(
                        tma_atom_a,
                        tAgA_mkl[(None, producer_state.count)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_nkl[(None, producer_state.count)],
                        tBsB[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )

                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(producer_state)

        # ---- CONSUMER (MMA warp groups) ----
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self._mma_register_requirement)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, S
            )
            consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, S
            )

            num_k_blocks = cute.size(tCrA, mode=[2])
            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx

                consumer_read_state.reset_count()
                consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                # Prologue MMA
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None, None, k_block_idx,
                            consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    consumer_read_state.advance()

                # Mainloop
                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None, None, k_block_idx,
                            consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(consumer_release_state)
                    consumer_release_state.advance()
                    consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(consumer_release_state)
                    consumer_release_state.advance()

                # Epilogue: write this tile to gmem
                tCgC_slice = tCgC[(None, None, None,
                                   tile_coord_mnl[0], tile_coord_mnl[1])]
                epilogue_f32 = alpha * accumulators.load() + beta * tCgC_slice.load()
                tCrC = cute.make_fragment_like(tCgC_slice)
                tCrC.store(epilogue_f32.to(cutlass.Float16))
                cute.autovec_copy(tCrC, tCgC_slice)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def run_gemm_cluster(M=512, N=512, K=512,
                     bm=128, bn=256,
                     num_consumer_warpgroups=2,
                     num_stages=4,
                     cluster_m=2, cluster_n=1):
    """Run the cluster multicast HGEMM and verify against NumPy."""
    np.random.seed(42)

    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))
    C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float16))

    A_d = cp.array(A_h, order="F")
    B_d = cp.array(B_h, order="F")
    C_d = cp.array(C_h, order="F")

    A_t = from_dlpack(A_d.T, assumed_align=16)
    B_t = from_dlpack(B_d.T, assumed_align=16)
    C_t = from_dlpack(C_d, assumed_align=16)

    alpha, beta = 1.0, 0.0
    gemm = HgemmCluster(
        bm=bm, bn=bn,
        num_consumer_warpgroups=num_consumer_warpgroups,
        num_stages=num_stages,
        cluster_m=cluster_m, cluster_n=cluster_n,
    )
    gemm(A_t, B_t, C_t, alpha=alpha, beta=beta)
    cp.cuda.Device().synchronize()

    D_ref = alpha * (A_h.T.astype(np.float32) @ B_h.astype(np.float32)) \
            + beta * C_h.astype(np.float32)
    D_out = cp.asnumpy(C_d).astype(np.float32)

    abs_err = float(np.max(np.abs(D_out - D_ref)))
    rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
    passed = bool(np.allclose(D_out, D_ref, atol=5e-2, rtol=5e-2))

    status = "PASS" if passed else "FAIL"
    print(f"hgemm_cluster ({M}x{N}x{K}): {status}  "
          f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}")
    return passed


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N = int(sys.argv[2]) if len(sys.argv) > 2 else M
    K = int(sys.argv[3]) if len(sys.argv) > 3 else M
    ok = run_gemm_cluster(M, N, K)
    sys.exit(0 if ok else 1)
