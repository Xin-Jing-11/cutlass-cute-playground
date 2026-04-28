# Instantiates CuTeDSL HGEMM variants, mirroring CUTLASS/hgemm/instantiate.cu
# Best configs from autotuning at 4096x4096x4096 on H100 NVL.
#
# Each entry: variant_name -> (class, kwargs)

from CuTeDSL.hgemm.hgemm_wgmma_tma import HgemmWgmmaTma
from CuTeDSL.hgemm.hgemm_warp_specialization import HgemmWarpSpecialization
from CuTeDSL.hgemm.hgemm_persistent import HgemmPersistent
from CuTeDSL.hgemm.hgemm_cluster import HgemmCluster
from CuTeDSL.hgemm.hgemm_epilogue import HgemmEpilogue
from CuTeDSL.hgemm.hgemm_optimized import HgemmOptimized

VARIANTS = {
    # Best: 128x128x64x3 (0.366 ms, 376K GF/s)
    "wgmma_tma_128x128x64x3":     (HgemmWgmmaTma, dict(bm=128, bn=128, num_stages=3)),
    # "wgmma_tma_64x256x64x3":    (HgemmWgmmaTma, dict(bm=64, bn=256, num_stages=3)),  # 0.366 ms — tied

    # Best: 128x256x64x3 (0.326 ms, 421K GF/s)
    "warp_spec_128x256x64x3":     (HgemmWarpSpecialization, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=3)),
    # "warp_spec_128x256x64x4":   (HgemmWarpSpecialization, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=4)),  # 0.328 ms

    # Best: 128x256x64x4 (0.327 ms, 421K GF/s)
    "persistent_128x256x64x4":    (HgemmPersistent, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=4)),
    # "persistent_128x256x64x3":  (HgemmPersistent, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=3)),  # 0.330 ms

    # Best: 128x256x64x3 (0.317 ms, 434K GF/s)
    "cluster_128x256x64x3":       (HgemmCluster, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=3, cluster_m=2, cluster_n=1)),
    # "cluster_128x256x64x4":     (HgemmCluster, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=4, cluster_m=2, cluster_n=1)),  # 0.322 ms

    # Best: 128x256x64x2 (0.301 ms, 457K GF/s)
    "epilogue_128x256x64x2":      (HgemmEpilogue, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=2, cluster_m=2, cluster_n=1)),
    # "epilogue_128x256x64x3":    (HgemmEpilogue, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=3, cluster_m=2, cluster_n=1)),  # 0.302 ms

    # Best: 128x256x64x2 (0.300 ms, 458K GF/s)
    "optimized_128x256x64x2":     (HgemmOptimized, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=2, cluster_m=2, cluster_n=1, swizzle_size=4)),
    # "optimized_128x256x64x3":   (HgemmOptimized, dict(bm=128, bn=256, num_consumer_warpgroups=2, num_stages=3, cluster_m=2, cluster_n=1, swizzle_size=4)),  # 0.301 ms
}
