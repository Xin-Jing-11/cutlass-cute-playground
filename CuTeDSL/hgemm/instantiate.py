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

VARIANTS = {
    # mma: SM80 16x8x16 tensor core, 128-bit g2s, 3-tensor interface (A, B, C in-place)
    "mma_128x128x32":      (HgemmMma, dict(bm=128, bn=128, bk=32)),
}
