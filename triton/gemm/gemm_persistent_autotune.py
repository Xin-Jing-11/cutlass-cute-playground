#!/usr/bin/env python3
"""Autotune driver for the persistent Triton GEMM kernels in gemm_persistent.py.

Sweeps a Cartesian product over tile / warp / stage / swizzle for both
non-TMA and TMA persistent kernels, persists the chosen configs to JSON, and
prints a comparison vs cuBLAS.

After running, copy the printed best configs into _BEST_PERSISTENT and
_BEST_TMA_PERSISTENT in gemm_persistent.py.
"""

import json
import pathlib
import time

import torch
import triton

from gemm_persistent import (
    matmul_kernel_persistent_jit,
    matmul_kernel_tma_persistent_jit,
    _tma_set_block_size_hook,
)
from triton.tools.tensor_descriptor import TensorDescriptor


# ---------------------------------------------------------------------------
# Config sweep
# ---------------------------------------------------------------------------
def _make_configs(pre_hook=None):
    return [
        triton.Config(
            {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn, 'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': gs},
            num_warps=nw, num_stages=ns,
            pre_hook=pre_hook,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [32, 64]
        for gs in [1, 4, 8]
        for nw in [4, 8]
        for ns in [3, 4]
        if bm * bn // (nw * 32) <= 256
    ]


_AUTOTUNE_CONFIGS_PERSISTENT = _make_configs()
_AUTOTUNE_CONFIGS_TMA        = _make_configs(pre_hook=_tma_set_block_size_hook)


# ---------------------------------------------------------------------------
# Persistent JSON cache (one file with two top-level sections)
# ---------------------------------------------------------------------------
_CACHE_PATH = pathlib.Path(__file__).with_name("gemm_persistent.autotune.json")


def _config_to_dict(cfg):
    return {
        "kwargs":     dict(cfg.kwargs),
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
        "num_ctas":   getattr(cfg, "num_ctas", 1),
        "maxnreg":    getattr(cfg, "maxnreg", None),
    }


def _config_from_dict(d, pre_hook=None):
    return triton.Config(
        d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d.get("num_ctas", 1),
        maxnreg=d.get("maxnreg"),
        pre_hook=pre_hook,
    )


def _load_section(section, pre_hook=None):
    if not _CACHE_PATH.exists():
        return {}
    with open(_CACHE_PATH) as f:
        raw = json.load(f).get(section, {})
    return {tuple(int(x) for x in k.split("x")): _config_from_dict(v, pre_hook=pre_hook)
            for k, v in raw.items()}


def _save_all(persistent_cache, tma_cache):
    out = {
        "persistent": {f"{k[0]}x{k[1]}x{k[2]}": _config_to_dict(v)
                       for k, v in persistent_cache.items()},
        "tma_persistent": {f"{k[0]}x{k[1]}x{k[2]}": _config_to_dict(v)
                           for k, v in tma_cache.items()},
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Re-wrap the bare jit kernels with the wide config sweeps
# ---------------------------------------------------------------------------
matmul_kernel_persistent = triton.autotune(
    configs=_AUTOTUNE_CONFIGS_PERSISTENT, key=['M', 'N', 'K'],
)(matmul_kernel_persistent_jit)
matmul_kernel_persistent.cache.update(_load_section("persistent"))

matmul_kernel_tma_persistent = triton.autotune(
    configs=_AUTOTUNE_CONFIGS_TMA, key=['M', 'N', 'K'],
)(matmul_kernel_tma_persistent_jit)
matmul_kernel_tma_persistent.cache.update(
    _load_section("tma_persistent", pre_hook=_tma_set_block_size_hook))


def _save_caches():
    _save_all(matmul_kernel_persistent.cache, matmul_kernel_tma_persistent.cache)


# ---------------------------------------------------------------------------
# Host wrappers (auto-save after a new shape gets tuned)
# ---------------------------------------------------------------------------
def matmul_persistent(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = lambda meta: (
        min(NUM_SMS,
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])),
    )
    keys_before = set(matmul_kernel_persistent.cache)
    matmul_kernel_persistent[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
    )
    if set(matmul_kernel_persistent.cache) != keys_before:
        _save_caches()
    return c


def matmul_tma_persistent(a, b, warp_specialize=False, epilogue_subtile=False):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    b_TN = b.T.contiguous()
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Initial block shapes; pre_hook overrides per autotune trial
    init = _AUTOTUNE_CONFIGS_TMA[0].kwargs
    BM0, BN0, BK0 = init['BLOCK_SIZE_M'], init['BLOCK_SIZE_N'], init['BLOCK_SIZE_K']
    a_desc = TensorDescriptor.from_tensor(a,    [BM0, BK0])
    b_desc = TensorDescriptor.from_tensor(b_TN, [BN0, BK0])
    c_desc = TensorDescriptor.from_tensor(
        c, [BM0, BN0 // 2] if epilogue_subtile else [BM0, BN0])

    triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device='cuda'))

    grid = lambda meta: (
        min(NUM_SMS,
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])),
    )
    keys_before = set(matmul_kernel_tma_persistent.cache)
    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        FP8_OUTPUT=False,
        EPILOGUE_SUBTILE=epilogue_subtile,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=warp_specialize,
    )
    if set(matmul_kernel_tma_persistent.cache) != keys_before:
        _save_caches()
    return c


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------
def _bench_cuda_events(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    import argparse
    p = argparse.ArgumentParser(description="Persistent Triton GEMM autotune + cuBLAS comparison")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    M, N, K = args.m, args.n, args.k
    cached_p = (M, N, K) in matmul_kernel_persistent.cache
    cached_t = (M, N, K) in matmul_kernel_tma_persistent.cache
    print(f"sweep size: persistent={len(_AUTOTUNE_CONFIGS_PERSISTENT)}, tma={len(_AUTOTUNE_CONFIGS_TMA)}")
    print(f"problem shape: M={M}, N={N}, K={K}")
    print(f"cache hits: persistent={cached_p}, tma_persistent={cached_t}  ({_CACHE_PATH.name})")

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c_ref = a @ b
    flop = 2 * M * N * K

    # First calls: trigger autotune (or use cached choice)
    t0 = time.perf_counter(); matmul_persistent(a, b); torch.cuda.synchronize()
    print(f"first persistent call ({'cached' if cached_p else 'autotune'}): {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter(); matmul_tma_persistent(a, b); torch.cuda.synchronize()
    print(f"first tma_persistent call ({'cached' if cached_t else 'autotune'}): {time.perf_counter()-t0:.1f}s")

    print(f"\nbest persistent:     {matmul_kernel_persistent.best_config}")
    print(f"best tma_persistent: {matmul_kernel_tma_persistent.best_config}\n")

    runners = {
        'cuBLAS (torch.matmul)':           lambda: torch.matmul(a, b),
        'persistent (autotuned)':          lambda: matmul_persistent(a, b),
        'tma_persistent (autotuned)':      lambda: matmul_tma_persistent(a, b),
        'tma_persistent + epi_subtile':    lambda: matmul_tma_persistent(a, b, epilogue_subtile=True),
    }

    print(f"{'kernel':<35}  {'time(ms)':>10}  {'GF/s':>9}  {'%cuBLAS':>9}")
    print("-" * 70)
    cublas_ms = _bench_cuda_events(runners['cuBLAS (torch.matmul)'], args.warmup, args.iters)
    print(f"{'cuBLAS (torch.matmul)':<35}  {cublas_ms:>10.4f}  {flop/cublas_ms/1e9*1e3:>9.0f}  {100.0:>8.1f}%")
    for name, fn in runners.items():
        if name.startswith('cuBLAS'):
            continue
        ms = _bench_cuda_events(fn, args.warmup, args.iters)
        print(f"{name:<35}  {ms:>10.4f}  {flop/ms/1e9*1e3:>9.0f}  {100*cublas_ms/ms:>8.1f}%")


if __name__ == "__main__":
    main()
