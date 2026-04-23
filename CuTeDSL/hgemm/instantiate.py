# Instantiates CuTeDSL HGEMM variants, mirroring CUTLASS/hgemm/instantiate.cu
#
# Each entry: variant_name -> (class, kwargs)

from CuTeDSL.hgemm.hgemm_mma import HgemmMma
from CuTeDSL.hgemm.hgemm_multistage import HgemmMultistage
from CuTeDSL.hgemm.hgemm_wgmma import HgemmWgmma

VARIANTS = {
    # Best config per family from autotuning at 4096³ on H100 NVL
    "mma_128x128x16":              (HgemmMma, dict(bm=128, bn=128, bk=16)),
    "multistage_128x128x32x3":     (HgemmMultistage, dict(bm=128, bn=128, bk=32, num_stages=3)),
    "wgmma_128x128x64x3":         (HgemmWgmma, dict(bm=128, bn=128, num_stages=3)),
}
