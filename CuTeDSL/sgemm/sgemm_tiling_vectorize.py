"""
Vectorize SGEMM using CuTe DSL (TN): D = alpha * A^T * B + beta * C

128-bit (float4) vectorized gmem→smem copies, with a `Swizzle<B, 2, S>` XOR
swizzle on sA. The M=2 offset leaves the low 2 bits of the flat index untouched,
preserving 4-float alignment so 128-bit stores remain valid.

No MC variant: A gmem is (M,K):(K,1) so K is contiguous, but MC smem has M
contiguous — the layout mismatch prevents vectorized stores.

A stored (K,M) col-major → CuTe (M,K):(K,1).
B stored (K,N) col-major → CuTe (N,K):(K,1).
C stored (M,N) col-major → CuTe (M,N):(1,M).
Float32 in/out.
"""

import math

import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
from cutlass.cute import nvgpu


class SgemmTilingVectorize:
    """
    Vectorize SGEMM: 128-bit gmem→smem copies, no swizzle.
    (BM/TM) x (BN/TN) threads per CTA.
    Each thread computes TM x TN output elements.
    """

    def __init__(self, bm=128, bn=128, bk=16, tm=8, tn=8):
        self._bm = bm
        self._bn = bn
        self._bk = bk
        self._tm = tm
        self._tn = tn
        assert bm % tm == 0 and bn % tn == 0
        assert bk % 4 == 0, "BK must be divisible by 4 for 128-bit vectorization"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M, K):(K,1) K-contiguous, f32
        mB: cute.Tensor,  # (N, K):(K,1) K-contiguous, f32
        mC: cute.Tensor,  # (M, N):(1,M) col-major, f32
        alpha: cutlass.Float32 = 1.0,
        beta: cutlass.Float32 = 0.0,
        stream: cuda_driver.CUstream = cuda_driver.CUstream(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        BM, BN, BK = self._bm, self._bn, self._bk
        TM, TN = self._tm, self._tn
        Tm, Tn = BM // TM, BN // TN
        num_threads = Tm * Tn
        VEC = 4  # 128-bit / 32-bit = 4 floats
        BK_VEC = BK // VEC

        # --- sA smem: Swizzle<B,2,S> to preserve 128-bit alignment ---
        atom_M = max(BK, Tm)
        swz_M = 2                              # preserve low 2 bits (4-float alignment)
        swz_B = int(math.log2(BK)) - swz_M     # remaining k-bits to XOR
        swz_S = int(math.log2(atom_M))
        assert swz_B > 0, "BK must be > VEC=4 for M=2 swizzle"
        swizzle_atom = cute.make_composed_layout(
            cute.make_swizzle(swz_B, swz_M, swz_S), 0,
            cute.make_layout((atom_M, BK), stride=(BK, 1)))
        sA_layout = cute.tile_to_shape(swizzle_atom, (BM, BK), (0, 1))
        sB_layout = cute.make_layout((BN, BK), stride=(BK, 1))

        # --- gmem→smem: vectorized 128-bit copies ---
        # Copy atom: 128 bits = 4 floats per copy
        vec_copy_atom = cute.make_copy_atom(
            nvgpu.CopyUniversalOp(), mA.element_type,
            num_bits_per_copy=128)

        # Thread layout tiles (M_threads, K_threads) over the (BM, BK/VEC) space
        thr_k = BK_VEC
        thr_m = num_threads // thr_k
        g2s_thr_layout = cute.make_layout((thr_m, thr_k), stride=(thr_k, 1))
        g2s_val_layout = cute.make_layout((1, VEC))  # 1 along M, VEC along K

        tiled_copy_A = cute.make_tiled_copy_tv(
            vec_copy_atom, g2s_thr_layout, g2s_val_layout)
        tiled_copy_B = cute.make_tiled_copy_tv(
            vec_copy_atom, g2s_thr_layout, g2s_val_layout)

        # --- tiled_mma: scalar FMA across (Tm, Tn) threads ---
        mma_op = cute.make_mma_atom(nvgpu.MmaUniversalOp(cutlass.Float32))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((Tm, Tn, 1), stride=(1, Tm, 0)))

        # --- smem→register: scalar copies aligned with MMA ---
        s2r_atom = cute.make_copy_atom(
            nvgpu.CopyUniversalOp(), cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width)
        s2r_copy_A = cute.make_tiled_copy_A(s2r_atom, tiled_mma)
        s2r_copy_B = cute.make_tiled_copy_B(s2r_atom, tiled_mma)

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
        sA_layout: cute.ComposedLayout,
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
        sA = smem.allocate_tensor(cutlass.Float32, sA_layout)
        sB = smem.allocate_tensor(cutlass.Float32, sB_layout)

        # --- gmem→smem via vectorized tiled_copy ---
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

        # --- smem→register copies ---
        thr_s2r_a = s2r_A.get_slice(tidx)
        tXsA = thr_s2r_a.partition_S(sA)    # (CPY, MMA_M, MMA_K)
        tXrA = thr_s2r_a.retile(tCrA)       # (CPY, MMA_M, MMA_K)

        thr_s2r_b = s2r_B.get_slice(tidx)
        tXsB = thr_s2r_b.partition_S(sB)    # (CPY, MMA_N, MMA_K)
        tXrB = thr_s2r_b.retile(tCrB)       # (CPY, MMA_N, MMA_K)

        # Mainloop
        k_max = cute.size(tAgA, mode=[3])
        k_block_max = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_max):
            # gmem → smem (vectorized 128-bit copies)
            cute.copy(g2s_A, tAgA[None, None, None, k_tile], tAsA)
            cute.copy(g2s_B, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.sync_threads()
            # inner K-loop: smem→register, then gemm
            for k_block in cutlass.range(k_block_max, unroll_full=True):
                cute.copy(s2r_A, tXsA[None, None, k_block], tXrA[None, None, k_block])
                cute.copy(s2r_B, tXsB[None, None, k_block], tXrB[None, None, k_block])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                          tCrB[None, None, k_block], tCrC)
            cute.arch.sync_threads()

        # Epilogue: C = alpha * acc + beta * C
        epilogue_op = lambda acc: alpha * acc + beta * tCgC.load()
        tCgC.store(epilogue_op(tCrC.load()))
