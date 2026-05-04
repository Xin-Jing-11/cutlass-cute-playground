#!/usr/bin/env python3
"""Autotune driver for the Triton FP16 GEMM kernels defined in gemm.py.

Sweeps a Cartesian product over tile / warp / stage / swizzle for both:
  - the cp.async kernel (matmul_kernel_jit)
  - the TMA kernel (matmul_kernel_tma_jit), with WARP_SPECIALIZE on AND off

Persists chosen configs to a JSON sidecar so subsequent runs skip the search.

Use this when:
  - Tuning for a new GPU
  - Tuning for a new shape
  - Verifying the hardcoded configs in gemm.py are still optimal

After running, copy the printed best configs into _BEST_CONFIG / _BEST_TMA_CONFIG
in gemm.py.
"""

import json
import pathlib
import time

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from gemm import (
    DEVICE,
    matmul_kernel_jit,
    matmul_kernel_tma_jit,
    _tma_set_block_size_hook,
)


# ---------------------------------------------------------------------------
# Config sweeps (one shared sweep grid, with a TMA flavor that adds pre_hook)
# ---------------------------------------------------------------------------
def _make_configs(pre_hook=None):
    return [
        triton.Config(
            {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn,
             'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': gs},
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


_AUTOTUNE_CONFIGS     = _make_configs()
_AUTOTUNE_CONFIGS_TMA = _make_configs(pre_hook=_tma_set_block_size_hook)


# ---------------------------------------------------------------------------
# Persistent JSON cache (one file with multiple top-level sections)
# ---------------------------------------------------------------------------
_CACHE_PATH = pathlib.Path(__file__).with_name("gemm.autotune.json")


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


# Cache keys can contain torch.dtype objects (Triton's specialization info).
# Round-trip those through their str() form via this lookup table.
_DTYPE_BY_NAME = {str(d): d for d in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int8, torch.int16, torch.int32, torch.int64, torch.bool,
)}


def _encode_part(x):
    return str(x)


def _decode_part(s):
    if s in _DTYPE_BY_NAME:
        return _DTYPE_BY_NAME[s]
    if s in ("True", "False"):
        return s == "True"
    if s.lstrip("-").isdigit():
        return int(s)
    return s  # fall back to raw string


def _key_to_str(key):
    return "|".join(_encode_part(x) for x in key)


def _key_from_str(s):
    return tuple(_decode_part(x) for x in s.split("|"))


def _load_section(section, pre_hook=None):
    if not _CACHE_PATH.exists():
        return {}
    with open(_CACHE_PATH) as f:
        raw = json.load(f).get(section, {})
    return {_key_from_str(k): _config_from_dict(v, pre_hook=pre_hook)
            for k, v in raw.items()}


def _save_all_sections():
    out = {
        "matmul": {_key_to_str(k): _config_to_dict(v)
                   for k, v in matmul_kernel.cache.items()},
        "matmul_tma": {_key_to_str(k): _config_to_dict(v)
                       for k, v in matmul_kernel_tma.cache.items()},
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Re-wrap both bare jit kernels with the wide config sweeps
# ---------------------------------------------------------------------------
matmul_kernel = triton.autotune(
    configs=_AUTOTUNE_CONFIGS, key=['M', 'N', 'K'],
)(matmul_kernel_jit)
matmul_kernel.cache.update(_load_section("matmul"))

matmul_kernel_tma = triton.autotune(
    configs=_AUTOTUNE_CONFIGS_TMA, key=['M', 'N', 'K', 'WARP_SPECIALIZE'],
)(matmul_kernel_tma_jit)
matmul_kernel_tma.cache.update(_load_section("matmul_tma", pre_hook=_tma_set_block_size_hook))


# ---------------------------------------------------------------------------
# Host wrappers (auto-save after a new shape gets tuned)
# ---------------------------------------------------------------------------
def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    keys_before = set(matmul_kernel.cache)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    if set(matmul_kernel.cache) != keys_before:
        _save_all_sections()
    return c


def matmul_tma(a, b, warp_specialize=False):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    b_TN = b.T.contiguous()

    init = _AUTOTUNE_CONFIGS_TMA[0].kwargs
    BM0, BN0, BK0 = init['BLOCK_SIZE_M'], init['BLOCK_SIZE_N'], init['BLOCK_SIZE_K']
    a_desc = TensorDescriptor.from_tensor(a,    [BM0, BK0])
    b_desc = TensorDescriptor.from_tensor(b_TN, [BN0, BK0])
    c_desc = TensorDescriptor.from_tensor(c,    [BM0, BN0])

    triton.set_allocator(
        lambda size, align, _: torch.empty(size, dtype=torch.int8, device='cuda'))

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    keys_before = set(matmul_kernel_tma.cache)
    matmul_kernel_tma[grid](
        a_desc, b_desc, c_desc, M, N, K,
        WARP_SPECIALIZE=warp_specialize,
    )
    if set(matmul_kernel_tma.cache) != keys_before:
        _save_all_sections()
    return c


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------
def _bench_cuda_events(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    import argparse
    p = argparse.ArgumentParser(description="Triton GEMM autotune (cp.async + TMA + WS) + cuBLAS comparison")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"sweeps: cp.async={len(_AUTOTUNE_CONFIGS)}, TMA={len(_AUTOTUNE_CONFIGS_TMA)} configs")
    print(f"problem shape: M={M}, N={N}, K={K}")
    print(f"cache hits:")
    print(f"  matmul         (M,N,K):     {(M,N,K) in matmul_kernel.cache}")
    print(f"  matmul_tma     (M,N,K,ws=F): {(M,N,K,False) in matmul_kernel_tma.cache}")
    print(f"  matmul_tma     (M,N,K,ws=T): {(M,N,K,True)  in matmul_kernel_tma.cache}")

    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c_ref = a @ b
    flop = 2 * M * N * K

    runners = [
        ("cp.async (autotuned)",      lambda: matmul(a, b)),
        ("TMA (autotuned)",           lambda: matmul_tma(a, b, warp_specialize=False)),
        ("TMA + WS (autotuned)",      lambda: matmul_tma(a, b, warp_specialize=True)),
    ]

    # First call: triggers autotune for any shape not yet cached
    for name, fn in runners:
        t0 = time.perf_counter()
        c = fn()
        torch.cuda.synchronize()
        rel = (c - c_ref).abs().max().item() / max(c_ref.abs().max().item(), 1e-9)
        print(f"  {name:<22} first call: {time.perf_counter()-t0:6.1f}s  rel_err={rel:.2e}")

    print()
    print(f"best matmul:           {matmul_kernel.best_config}")
    print(f"best matmul_tma (ws=F/T): inspect matmul_kernel_tma.cache")
    print()

    cublas_ms = _bench_cuda_events(lambda: torch.matmul(a, b), args.warmup, args.iters)
    print(f"{'kernel':<28}  {'time(ms)':>10}  {'GF/s':>9}  {'%cuBLAS':>9}")
    print("-" * 62)
    print(f"{'cuBLAS (torch.matmul)':<28}  {cublas_ms:>10.4f}  {flop/cublas_ms/1e9*1e3:>9.0f}  {100.0:>8.1f}%")
    for name, fn in runners:
        ms = _bench_cuda_events(fn, args.warmup, args.iters)
        print(f"{'Triton ' + name:<28}  {ms:>10.4f}  {flop/ms/1e9*1e3:>9.0f}  {100*cublas_ms/ms:>8.1f}%")


if __name__ == "__main__":
    main()
