# Instantiates CuTeDSL HGEMM variants, mirroring CUTLASS/hgemm/instantiate.cu
# (3-file layout; CuTeDSL has no TMA kernel so only 2 variants ship).
#
# Each entry: variant_name -> (class, kwargs)

from CuTeDSL.hgemm.hgemm_mma import HgemmMma
from CuTeDSL.hgemm.hgemm_multistage import HgemmMultistage

VARIANTS = {
    # mma: SM80 16x8x16 tensor core + ldmatrix s2r, swizzled smem
    "mma_64x256x32":           (HgemmMma,        dict(bm=64, bn=256, bk=32)),

    # multistage: cp.async pipelined gmem→smem + ldmatrix s2r
    "multistage_128x128x32x3": (HgemmMultistage, dict(bm=128, bn=128, bk=32, num_stages=3)),
}
