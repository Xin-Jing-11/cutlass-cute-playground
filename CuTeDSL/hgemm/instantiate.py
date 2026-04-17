# Instantiates CuTeDSL HGEMM variants with specific parameters,
# mirroring CUTLASS/hgemm/instantiate.cu.
#
# Each entry: variant_name -> (class, kwargs)
#
# NOTE: HgemmMma omits smem swizzle due to a CuTeDSL bug where
# ComposedLayout + multi-K s2r iteration (k_block_max > 1) produces
# wrong results. The C++ version uses Swizzle<2,3,6> on both sA and sB.
# Same root cause as NVIDIA/cutlass#3160.

from CuTeDSL.hgemm.hgemm_mma import HgemmMma
from CuTeDSL.hgemm.hgemm_mma_ldmatrix import HgemmMmaLdmatrix
from CuTeDSL.hgemm.hgemm_multistage import HgemmMultistage

VARIANTS = {
    # mma: SM80 16x8x16 tensor core, scalar s2r
    "mma_128x128x32":              (HgemmMma, dict(bm=128, bn=128, bk=32)),

    # mma_ldmatrix: same MMA, but with ldmatrix s2r
    "mma_ldmatrix_128x128x32":     (HgemmMmaLdmatrix, dict(bm=128, bn=128, bk=32)),

    # multistage: cp.async pipelined gmem→smem + ldmatrix s2r
    "multistage_128x128x32x2":    (HgemmMultistage, dict(bm=128, bn=128, bk=32, num_stages=2)),
    "multistage_128x128x32x3":    (HgemmMultistage, dict(bm=128, bn=128, bk=32, num_stages=3)),
    "multistage_128x128x32x4":    (HgemmMultistage, dict(bm=128, bn=128, bk=32, num_stages=4)),
}
