# Instantiates CuTeDSL SGEMM variants with specific parameters,
# mirroring CUDA/CUTLASS instantiate.cu.
#
# Each entry: variant_name -> (class, kwargs)
#
# NOTE: SgemmTiling with swizzle only works when Tn (= BN/TN) >= BK,
# so that the g2s/s2r tiled_copy covers the K dimension in one step.
# Smaller tiles (e.g. 64x64 with BK=16, Tn=8) fail due to a CuTeDSL bug
# in partition_D/partition_S on ComposedLayout with multi-K-iteration.
# The same configuration works correctly in C++ CuTe.

from CuTeDSL.sgemm.sgemm_naive import SgemmNaive
from CuTeDSL.sgemm.sgemm_smem import SgemmSmem
from CuTeDSL.sgemm.sgemm_tiling import SgemmTiling
from CuTeDSL.sgemm.sgemm_vectorization import SgemmVectorize

VARIANTS = {
    # naive
    "naive":                        (SgemmNaive, dict(block_size=32)),

    # smem
    "smem":                         (SgemmSmem, dict(block_size=32)),
    "smem_mc":                      (SgemmSmem, dict(block_size=32, mc=True)),

    # tiling (only 128x128 with BK=16 — see NOTE above for why 64x64 is excluded)
    "tiling_64x64x8x8x8":           (SgemmTiling, dict(bm=64, bn=64, bk=8, tm=8, tn=8)),
    "tiling_128x128x16x8x8":        (SgemmTiling, dict(bm=128, bn=128, bk=16, tm=8, tn=8)),
    "tiling_mc_64x64x8x8x8":        (SgemmTiling, dict(bm=64, bn=64, bk=8, tm=8, tn=8, mc=True)),
    "tiling_mc_128x128x16x8x8":     (SgemmTiling, dict(bm=128, bn=128, bk=16, tm=8, tn=8, mc=True)),

    # vectorize (128-bit g2s, no swizzle, no MC — see sgemm_vectorization.py)
    "vectorize_64x64x16x8x8":       (SgemmVectorize, dict(bm=64, bn=64, bk=16, tm=8, tn=8)),
    "vectorize_128x128x16x8x8":     (SgemmVectorize, dict(bm=128, bn=128, bk=16, tm=8, tn=8)),
}
