#!/usr/bin/env python3
"""Autotune driver for the FP8 GEMM kernels in gemm_fp8.py.

Sweeps configs for both:
  - matmul_fp8        (per-tensor scaled FP8)
  - matmul_mxfp8      (block-scaled FP8 / MX format)

Persists chosen configs to a JSON sidecar. Compares against cuBLAS FP16.

Use this when:
  - You're tuning for a new GPU
  - You're tuning for a new shape
  - You want to verify the hardcoded configs in gemm_fp8.py are still optimal
"""

import json
import pathlib
import time

import torch
import triton

from gemm_fp8 import (
    DEVICE,
    MX_BLOCK_SIZE,
    matmul_fp8_kernel_jit,
    matmul_mxfp8_kernel_jit,
    quantize_per_tensor_fp8,
    quantize_mxfp8,
)


# ---------------------------------------------------------------------------
# Config sweeps
# ---------------------------------------------------------------------------
def _make_fp8_configs():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn,
             'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': gs},
            num_warps=nw, num_stages=ns,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for gs in [1, 4, 8]
        for nw in [4, 8]
        for ns in [3, 4]
        if bm * bn // (nw * 32) <= 256
    ]


def _make_mxfp8_configs():
    # MX requires BLOCK_SIZE_K to be a multiple of 32; we keep it >= 64
    return [
        triton.Config(
            {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn,
             'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': gs},
            num_warps=nw, num_stages=ns,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for gs in [1, 4, 8]
        for nw in [4, 8]
        for ns in [3, 4]
        if bm * bn // (nw * 32) <= 256 and bk % MX_BLOCK_SIZE == 0
    ]


_CONFIGS_FP8   = _make_fp8_configs()
_CONFIGS_MXFP8 = _make_mxfp8_configs()


# ---------------------------------------------------------------------------
# Persistent JSON cache (sections for each kernel)
# ---------------------------------------------------------------------------
_CACHE_PATH = pathlib.Path(__file__).with_name("gemm_fp8.autotune.json")


_DTYPE_BY_NAME = {str(d): d for d in (
    torch.float16, torch.bfloat16, torch.float32,
    torch.float8_e4m3fn, torch.float8_e5m2,
    torch.uint8,
)}


def _config_to_dict(cfg):
    return {
        "kwargs":     dict(cfg.kwargs),
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
        "num_ctas":   getattr(cfg, "num_ctas", 1),
        "maxnreg":    getattr(cfg, "maxnreg", None),
    }


def _config_from_dict(d):
    return triton.Config(
        d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d.get("num_ctas", 1),
        maxnreg=d.get("maxnreg"),
    )


def _decode_part(s):
    if s in _DTYPE_BY_NAME:
        return _DTYPE_BY_NAME[s]
    if s in ("True", "False"):
        return s == "True"
    if s.lstrip("-").isdigit():
        return int(s)
    return s


def _key_to_str(key):
    return "|".join(str(x) for x in key)


def _key_from_str(s):
    return tuple(_decode_part(x) for x in s.split("|"))


def _load_section(section):
    if not _CACHE_PATH.exists():
        return {}
    with open(_CACHE_PATH) as f:
        raw = json.load(f).get(section, {})
    return {_key_from_str(k): _config_from_dict(v) for k, v in raw.items()}


def _save_all_sections():
    out = {
        "fp8":   {_key_to_str(k): _config_to_dict(v)
                  for k, v in matmul_fp8_kernel.cache.items()},
        "mxfp8": {_key_to_str(k): _config_to_dict(v)
                  for k, v in matmul_mxfp8_kernel.cache.items()},
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Re-wrap the bare jit kernels with the wide config sweeps
# ---------------------------------------------------------------------------
matmul_fp8_kernel = triton.autotune(
    configs=_CONFIGS_FP8, key=['M', 'N', 'K'],
)(matmul_fp8_kernel_jit)
matmul_fp8_kernel.cache.update(_load_section("fp8"))

matmul_mxfp8_kernel = triton.autotune(
    configs=_CONFIGS_MXFP8, key=['M', 'N', 'K'],
)(matmul_mxfp8_kernel_jit)
matmul_mxfp8_kernel.cache.update(_load_section("mxfp8"))


# ---------------------------------------------------------------------------
# Host wrappers (auto-save on new shape)
# ---------------------------------------------------------------------------
def matmul_fp8(a_fp8, b_fp8, sa=1.0, sb=1.0):
    M, K = a_fp8.shape
    K, N = b_fp8.shape
    c = torch.empty((M, N), device=a_fp8.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    keys_before = set(matmul_fp8_kernel.cache)
    matmul_fp8_kernel[grid](
        a_fp8, b_fp8, c, M, N, K,
        float(sa), float(sb),
        a_fp8.stride(0), a_fp8.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        c.stride(0), c.stride(1),
    )
    if set(matmul_fp8_kernel.cache) != keys_before:
        _save_all_sections()
    return c


def matmul_mxfp8(a_fp8, b_fp8_TN, a_scale, b_scale):
    M, K = a_fp8.shape
    N, K2 = b_fp8_TN.shape
    assert K == K2
    c = torch.empty((M, N), device=a_fp8.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    keys_before = set(matmul_mxfp8_kernel.cache)
    matmul_mxfp8_kernel[grid](
        a_fp8, b_fp8_TN, c,
        a_scale, b_scale,
        M, N, K,
        a_fp8.stride(0),    a_fp8.stride(1),
        b_fp8_TN.stride(1), b_fp8_TN.stride(0),
        c.stride(0),        c.stride(1),
        a_scale.stride(0),  a_scale.stride(1),
        b_scale.stride(0),  b_scale.stride(1),
    )
    if set(matmul_mxfp8_kernel.cache) != keys_before:
        _save_all_sections()
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
    p = argparse.ArgumentParser(description="FP8 + MXFP8 autotune + cuBLAS comparison")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"sweeps: fp8={len(_CONFIGS_FP8)}, mxfp8={len(_CONFIGS_MXFP8)} configs")
    print(f"problem: M={M}, N={N}, K={K}")
    cached_pt = any(k[:3] == (M, N, K) for k in matmul_fp8_kernel.cache)
    cached_mx = any(k[:3] == (M, N, K) for k in matmul_mxfp8_kernel.cache)
    print(f"cache: fp8={cached_pt}, mxfp8={cached_mx}  ({_CACHE_PATH.name})")

    torch.manual_seed(0)
    a_f16 = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b_f16 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c_fp16 = torch.matmul(a_f16, b_f16)
    rms_ref = c_fp16.float().pow(2).mean().sqrt().item()
    flop = 2 * M * N * K

    # Per-tensor FP8
    a_fp8_pt, sa = quantize_per_tensor_fp8(a_f16)
    b_fp8_pt, sb = quantize_per_tensor_fp8(b_f16)
    sa, sb = sa.item(), sb.item()

    # MXFP8 (B in [N, K])
    a_fp8_mx, a_s = quantize_mxfp8(a_f16)
    b_fp8_mx, b_s = quantize_mxfp8(b_f16.T.contiguous())

    # First call: trigger autotune for any uncached shape
    t0 = time.perf_counter(); c_pt = matmul_fp8(a_fp8_pt, b_fp8_pt, sa, sb); torch.cuda.synchronize()
    print(f"first FP8 call ({'cached' if cached_pt else 'autotune'}): {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter(); c_mx = matmul_mxfp8(a_fp8_mx, b_fp8_mx, a_s, b_s); torch.cuda.synchronize()
    print(f"first MXFP8 call ({'cached' if cached_mx else 'autotune'}): {time.perf_counter()-t0:.1f}s")

    rms_pt = (c_pt.float() - c_fp16.float()).pow(2).mean().sqrt().item() / rms_ref
    rms_mx = (c_mx.float() - c_fp16.float()).pow(2).mean().sqrt().item() / rms_ref

    print(f"\nbest fp8:    {matmul_fp8_kernel.best_config}")
    print(f"best mxfp8:  {matmul_mxfp8_kernel.best_config}\n")
    print(f"correctness (RMS err / RMS ref):  fp8={rms_pt:.3e}  mxfp8={rms_mx:.3e}\n")

    # Bench
    fp16_ms = _bench_cuda_events(lambda: torch.matmul(a_f16, b_f16),                  args.warmup, args.iters)
    pt_ms   = _bench_cuda_events(lambda: matmul_fp8(a_fp8_pt, b_fp8_pt, sa, sb),       args.warmup, args.iters)
    mx_ms   = _bench_cuda_events(lambda: matmul_mxfp8(a_fp8_mx, b_fp8_mx, a_s, b_s),   args.warmup, args.iters)

    print(f"{'kernel':<28}  {'time(ms)':>10}  {'TF/s':>8}  {'%cuBLAS-FP16':>14}")
    print("-" * 68)
    print(f"{'cuBLAS FP16 (torch.matmul)':<28}  {fp16_ms:>10.4f}  {flop/fp16_ms/1e12*1e3:>8.1f}  {100.0:>13.1f}%")
    print(f"{'Triton FP8 (per-tensor)':<28}  {pt_ms:>10.4f}  {flop/pt_ms/1e12*1e3:>8.1f}  {100*fp16_ms/pt_ms:>13.1f}%")
    print(f"{'Triton MXFP8 (block, E4M3)':<28}  {mx_ms:>10.4f}  {flop/mx_ms/1e12*1e3:>8.1f}  {100*fp16_ms/mx_ms:>13.1f}%")


if __name__ == "__main__":
    main()
