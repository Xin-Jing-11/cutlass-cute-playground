"""
Persistent warp-specialized cluster HGEMM with TMA store epilogue (TN):
    C = alpha * A^T * B

Instead of direct register-to-gmem writes (autovec_copy), accumulator results
go through shared memory (sC) and are written to gmem via TMA store using
PipelineTmaStore and StMatrix8x8x16bOp.

Layout convention (TN col-major):
    A stored (K,M) col-major  ->  CuTe tensor (M,K):(K,1)   K-contiguous
    B stored (K,N) col-major  ->  CuTe tensor (N,K):(K,1)   K-contiguous
    C stored (M,N) col-major  ->  CuTe tensor (M,N):(1,M)   M-contiguous
    Float16 in / Float32 accumulator / Float16 out.

BM=128, BN=256, BK=64, cluster=(2,1), 2 consumer warp-groups,
epilogue tile (128,32) with 4 epilogue stages.
"""

import sys
import math

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


class HgemmEpilogue:
    """Persistent warp-specialized cluster HGEMM with TMA store epilogue."""

    def __init__(
        self,
        bm: int = 128,
        bn: int = 256,
        num_consumer_warpgroups: int = 2,
        num_stages: int = 4,
        cluster_m: int = 2,
        cluster_n: int = 1,
    ):
        self._bm = bm
        self._bn = bn
        self._bk = 64
        self._num_consumer_warpgroups = num_consumer_warpgroups
        self._num_stages = num_stages
        self._cluster_shape_mn = (cluster_m, cluster_n)

        self._acc_dtype = cutlass.Float32

        self._atom_layout_mnk = (
            (2, 1, 1)
            if bm > 64 and bn > 128
            else (1, 1, 1)
        )
        self._num_mma_warp_groups = math.prod(self._atom_layout_mnk)
        self._num_dma_warp_groups = 1
        self._warps_per_wg = 4
        self._threads_per_wg = self._warps_per_wg * 32
        self._threads_per_cta = (
            (self._num_dma_warp_groups + self._num_mma_warp_groups)
            * self._threads_per_wg
        )

        self._load_warp_id = 0
        self._epi_store_warp_id = self._num_dma_warp_groups * self._warps_per_wg
        self._load_register_requirement = 40
        self._mma_register_requirement = 232

        self._occupancy = 1
        self._buffer_align_bytes = 1024
        self._smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self._epi_tile = (128, 32)
        self._epi_stage = 4

        self._num_mma_threads = (
            self._num_mma_warp_groups * self._threads_per_wg
        )
        self._epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self._num_mma_threads
        )

    # ------------------------------------------------------------------
    # Host-side JIT launcher
    # ------------------------------------------------------------------
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

        a_dtype = mA.element_type
        b_dtype = mB.element_type
        c_dtype = mC.element_type

        a_layout = utils.LayoutEnum.from_tensor(mA)
        b_layout = utils.LayoutEnum.from_tensor(mB)
        c_layout = utils.LayoutEnum.from_tensor(mC)

        # ---- Tiled MMA ----
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self._acc_dtype,
            self._atom_layout_mnk,
            tiler_mn=(64, BN),
        )
        mma_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        tile_shape_mnk = (BM, BN, mma_k * 4)

        cta_layout_mnk = cute.make_layout((*self._cluster_shape_mn, 1))

        # ---- Compute AB stages given epilogue smem footprint ----
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(self._epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * self._epi_stage
        mbar_bytes = 1024
        ab_stage = (
            self._smem_capacity // self._occupancy - (mbar_bytes + epi_bytes)
        ) // ab_bytes_per_stage

        # ---- Shared-memory layouts for A, B ----
        a_is_k_major = (
            a_layout.sm90_mma_major_mode()
            == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_is_k_major = (
            b_layout.sm90_mma_major_mode()
            == cute.nvgpu.warpgroup.OperandMajorMode.K
        )

        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_major_mode_size = tile_shape_mnk[2] if a_is_k_major else tile_shape_mnk[0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2] if b_is_k_major else tile_shape_mnk[1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        # ---- Epilogue shared-memory layout for C ----
        c_major_mode_size = (
            self._epi_tile[1] if c_layout.is_n_major_c() else self._epi_tile[0]
        )
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(self._epi_tile, self._epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        # ---- TMA load atoms (cluster-aware multicast) ----
        num_mcast_ctas_a = self._cluster_shape_mn[1]
        num_mcast_ctas_b = self._cluster_shape_mn[0]
        is_a_mcast = num_mcast_ctas_a > 1
        is_b_mcast = num_mcast_ctas_b > 1

        a_tma_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
            if is_a_mcast
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        )
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            a_tma_op, mA, a_smem_layout,
            (tile_shape_mnk[0], tile_shape_mnk[2]),
            num_multicast=num_mcast_ctas_a,
        )

        b_tma_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
            if is_b_mcast
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            b_tma_op, mB, b_smem_layout,
            (tile_shape_mnk[1], tile_shape_mnk[2]),
            num_multicast=num_mcast_ctas_b,
        )

        # ---- TMA store atom for epilogue ----
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mC, epi_smem_layout, self._epi_tile,
        )

        # ---- Persistent tile scheduler ----
        c_tile_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(mC, tiler=c_tile_shape)
        num_ctas_mn = gc[(0, (None, None))].shape
        num_ctas_mnl = (*num_ctas_mn, 1)
        cluster_shape_mnl = (*self._cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
            swizzle_size=1, raster_along_m=True,
        )
        max_active_clusters = 132
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters,
        )

        # ---- Shared storage ----
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                self._buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                self._buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[c_dtype, cute.cosize(epi_smem_layout_staged)],
                self._buffer_align_bytes,
            ]

        self._shared_storage = SharedStorage
        self._ab_stage = ab_stage
        self._c_layout = c_layout
        self._a_dtype = a_dtype
        self._b_dtype = b_dtype
        self._c_dtype = c_dtype
        self._is_a_mcast = is_a_mcast
        self._is_b_mcast = is_b_mcast
        self._tile_shape_mnk = tile_shape_mnk

        # ---- Launch kernel ----
        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            alpha, beta,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self._threads_per_cta, 1, 1],
            cluster=(*self._cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
        alpha: cutlass.Float32,
        beta: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        BM, BN, BK = self._tile_shape_mnk

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        # Cluster coordinate
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # Multicast masks
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self._is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self._is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self._a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self._b_dtype, b_smem_layout)

        # ---- Shared memory allocation ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # ---- Mainloop pipeline setup ----
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_mcast_ctas_a = self._cluster_shape_mn[1]
        num_mcast_ctas_b = self._cluster_shape_mn[0]
        mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self._num_mma_warp_groups * self._warps_per_wg
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self._ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        pipeline_init_arrive(
            cluster_shape_mn=self._cluster_shape_mn, is_relaxed=True
        )

        # ---- Smem tensors ----
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # ---- Global tile partitions ----
        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self._tile_shape_mnk, (None, 0, None)),
            (None, None),
        )
        gB_nk = cute.local_tile(
            mB_nk,
            cute.slice_(self._tile_shape_mnk, (0, None, None)),
            (None, None),
        )
        gC_mn = cute.local_tile(
            mC_mn,
            cute.slice_(self._tile_shape_mnk, (None, None, 0)),
            (None, None),
        )

        # ---- TMA load partitions (cluster-aware) ----
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )

        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2),
        )

        # ---- MMA thread partitions ----
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self._threads_per_wg
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self._num_mma_warp_groups, stride=self._threads_per_wg
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(
                warp_group_idx - self._num_dma_warp_groups
            )
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # Per-thread C partition for beta blending and accumulator shape
        mma_thread_idx = tidx - self._num_dma_warp_groups * self._threads_per_wg
        thr_mma_c = tiled_mma.get_slice(mma_thread_idx)
        tCgC = thr_mma_c.partition_C(gC_mn)
        gC_first_tile = gC_mn[(None, None, 0, 0)]
        tCgC_first = thr_mma_c.partition_C(gC_first_tile)
        acc_shape = tCgC_first.shape
        # Two accumulators: one for WGMMA (never retiled), one for R2S (retiled)
        accumulators = cute.make_rmem_tensor(acc_shape, self._acc_dtype)
        acc_r2s = cute.make_rmem_tensor(acc_shape, self._acc_dtype)

        k_tile_cnt = cute.size(gA_mk, mode=[3])

        # ---- Cluster sync ----
        pipeline_init_wait(cluster_shape_mn=self._cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self._num_dma_warp_groups

        # ============================================================
        # Producer (DMA warp group)
        # ============================================================
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self._load_register_requirement)

        if warp_idx == self._load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self._ab_stage
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mk = tAgA[(None, tile_coord_mnl[0], None)]
                tBgB_nk = tBgB[(None, tile_coord_mnl[1], None)]

                producer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(producer_state)

                    cute.copy(
                        tma_atom_a,
                        tAgA_mk[(None, producer_state.count)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_nk[(None, producer_state.count)],
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

        # ============================================================
        # Consumer (MMA warp groups)
        # ============================================================
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self._mma_register_requirement)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self._ab_stage
            )
            release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self._ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])
            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            # ---- R2S epilogue copy atoms ----
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self._c_layout,
                elem_ty_d=self._c_dtype,
                elem_ty_acc=self._acc_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self._c_layout.is_m_major_c(), 4,
                ),
                self._c_dtype,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                copy_atom_C, tiled_mma
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, tiled_copy_C_Atom,
            )

            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self._num_dma_warp_groups * self._threads_per_wg
            )
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(acc_r2s)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self._acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self._c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            # ---- TMA store pipeline ----
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self._num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self._epi_stage,
                producer_group=tma_store_producer_group,
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_slice = gC_mn[
                    (None, None, tile_coord_mnl[0], tile_coord_mnl[1])
                ]

                # ---- Mainloop ----
                read_state.reset_count()
                release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                # Prologue MMA
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        coord = (
                            None, None, k_block_idx, read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, accumulators,
                            tCrA[coord], tCrB[coord], accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    read_state.advance()

                # Steady-state MMA
                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        coord = (
                            None, None, k_block_idx, read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, accumulators,
                            tCrA[coord], tCrB[coord], accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                    read_state.advance()

                # Drain prologue releases
                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()

                # ---- Epilogue: R2S → TMA store ----
                # Note: TMA store epilogue only supports beta=0 (write-only)
                # Scale accumulators by alpha
                epilogue_f32 = alpha * accumulators.load()
                acc_r2s.store(epilogue_f32)

                tCgC_for_tma = cute.zipped_divide(gC_slice, self._epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1),
                    cute.group_modes(sC, 0, 2), tCgC_for_tma,
                )

                epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
                epi_tile_shape = tCgC_for_tma.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                num_prev_epi_tiles = (
                    tile_sched.num_tiles_executed * epi_tile_num
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Apply alpha * acc + beta * C per element
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[
                            epi_idx * size_tRS_rD + epi_v
                        ]

                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self._c_dtype))

                    epi_buffer = (
                        (num_prev_epi_tiles + epi_idx)
                        % cute.size(tRS_sD, mode=[3])
                    )
                    cute.copy(
                        tiled_copy_r2s, tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self._epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self._epi_store_warp_id:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                    self._epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def run_gemm_epilogue(M=512, N=512, K=512,
                      bm=128, bn=256,
                      num_consumer_warpgroups=2,
                      num_stages=4,
                      cluster_m=2, cluster_n=1):
    """Run and validate the TMA-store-epilogue HGEMM against NumPy."""
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

    gemm = HgemmEpilogue(
        bm=bm, bn=bn,
        num_consumer_warpgroups=num_consumer_warpgroups,
        num_stages=num_stages,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )
    gemm(A_t, B_t, C_t, alpha=1.0, beta=0.0)
    cp.cuda.Device().synchronize()

    D_ref = A_h.T.astype(np.float32) @ B_h.astype(np.float32)
    D_out = cp.asnumpy(C_d).astype(np.float32)

    abs_err = float(np.max(np.abs(D_out - D_ref)))
    rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
    passed = rel_err < 0.05

    status = "PASS" if passed else "FAIL"
    print(f"hgemm_epilogue ({M}x{N}x{K}): {status}  "
          f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}")
    return passed


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N = int(sys.argv[2]) if len(sys.argv) > 2 else M
    K = int(sys.argv[3]) if len(sys.argv) > 3 else M
    ok = run_gemm_epilogue(M, N, K)
    sys.exit(0 if ok else 1)
