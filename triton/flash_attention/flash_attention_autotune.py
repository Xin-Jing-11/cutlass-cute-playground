#!/usr/bin/env python3
"""Autotune driver for the Flash Attention kernel in flash_attention.py.

Sweeps a Cartesian product over BLOCK_M / BLOCK_N / num_warps / num_stages,
prunes configs that violate seq_len / D_MODEL divisibility, persists the
chosen config to JSON, and benchmarks vs PyTorch's SDPA (which dispatches
to FlashAttention-2 / cuDNN on supported GPUs).

After running, copy the printed best config into _BEST_FA_CONFIG in
flash_attention.py.
"""

import json
import pathlib
import time

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from flash_attention import (
    DEVICE,
    fused_attention_kernel_jit,
    _fa_set_block_size_hook,
)


# ---------------------------------------------------------------------------
# Sweep grid. Filtered for sanity in the comprehension; further pruned at
# autotune time based on seq_len/D_MODEL.
# ---------------------------------------------------------------------------
_FA_CONFIGS = [
    triton.Config(
        {'BLOCK_M': bm, 'BLOCK_N': bn},
        num_warps=nw, num_stages=ns,
        pre_hook=_fa_set_block_size_hook,
    )
    for bm in [64, 128]
    for bn in [32, 64, 128]
    for nw in [4, 8]
    for ns in [2, 3, 4]
    if bm >= bn          # Q-tile no smaller than K-tile (causal-loop alignment)
    and bm % bn == 0     # required so the diagonal sweep aligns to BLOCK_N
]


# Drop configs whose BLOCK_M/BLOCK_N can't divide seq_len, or whose BLOCK_N
# exceeds D_MODEL (the kernel asserts this statically).
def _fa_prune(configs, named_args, **kwargs):
    seq_len = named_args.get('seq_len', kwargs.get('seq_len'))
    d_model = kwargs.get('D_MODEL', named_args.get('D_MODEL'))
    if seq_len is None and d_model is None:
        return configs
    out = []
    for c in configs:
        BM, BN = c.kwargs['BLOCK_M'], c.kwargs['BLOCK_N']
        if seq_len is not None and (seq_len % BM or seq_len % BN):
            continue
        if d_model is not None and BN > d_model:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Persistent JSON cache
# ---------------------------------------------------------------------------
_CACHE_PATH = pathlib.Path(__file__).with_name("flash_attention.autotune.json")


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


_DTYPE_BY_NAME = {str(d): d for d in (
    torch.float16, torch.bfloat16, torch.float32,
)}


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


def _load_cache():
    if not _CACHE_PATH.exists():
        return {}
    with open(_CACHE_PATH) as f:
        raw = json.load(f)
    return {_key_from_str(k): _config_from_dict(v, pre_hook=_fa_set_block_size_hook)
            for k, v in raw.items()}


def _save_cache():
    raw = {_key_to_str(k): _config_to_dict(v)
           for k, v in fused_attention_kernel.cache.items()}
    with open(_CACHE_PATH, "w") as f:
        json.dump(raw, f, indent=2)


# ---------------------------------------------------------------------------
# Re-wrap the bare jit kernel with the wide sweep + pruning
# ---------------------------------------------------------------------------
fused_attention_kernel = triton.autotune(
    configs=_FA_CONFIGS,
    key=['seq_len', 'D_MODEL'],
    prune_configs_by={'early_config_prune': _fa_prune},
)(fused_attention_kernel_jit)
fused_attention_kernel.cache.update(_load_cache())


# ---------------------------------------------------------------------------
# Host wrapper (auto-saves the cache on a new shape)
# ---------------------------------------------------------------------------
def fused_attention(q, k, v, sm_scale):
    assert q.shape == k.shape == v.shape
    assert q.shape[-1] in {16, 32, 64, 128, 256}

    batch, head, seq_len, d_model = q.shape
    o = torch.empty_like(q)
    M = torch.empty((batch, head, seq_len), device=q.device, dtype=torch.float32)
    bhs = batch * head * seq_len

    # Initial block_shape — pre_hook will overwrite per autotune trial
    init = _FA_CONFIGS[0].kwargs
    BM0, BN0 = init['BLOCK_M'], init['BLOCK_N']
    desc_q = TensorDescriptor(q, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BM0, d_model])
    desc_k = TensorDescriptor(k, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BN0, d_model])
    desc_v = TensorDescriptor(v, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BN0, d_model])
    desc_o = TensorDescriptor(o, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BM0, d_model])

    triton.set_allocator(
        lambda size, align, _: torch.empty(size, dtype=torch.int8, device="cuda"))

    keys_before = set(fused_attention_kernel.cache)
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * head, 1)
    fused_attention_kernel[grid](
        M, batch, head,
        desc_q, desc_k, desc_v, desc_o, sm_scale, seq_len,
        D_MODEL=d_model,
    )
    if set(fused_attention_kernel.cache) != keys_before:
        _save_cache()
    return o


# ---------------------------------------------------------------------------
# Bench harness — vs torch SDPA (which uses FA-2 / cuDNN on supported GPUs)
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


def _attn_flops(B, H, N, D):
    # qk: 2*B*H*N*N*D ; pv: 2*B*H*N*N*D ; halve for causal (~upper triangle = 0)
    return 0.5 * 2 * 2 * B * H * N * N * D


def main():
    import argparse
    p = argparse.ArgumentParser(description="Triton FA autotune + comparison vs torch SDPA")
    p.add_argument("--b", type=int, default=4)
    p.add_argument("--h", type=int, default=16)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    B, H, N, D = args.b, args.h, args.n, args.d
    # Triton's cache key includes dtype-specialization parts beyond our `key=`
    # list, so check by prefix match on (seq_len, D_MODEL).
    cached = any(k[:2] == (N, D) for k in fused_attention_kernel.cache)
    print(f"sweep size: {len(_FA_CONFIGS)} configs (pre-prune)")
    print(f"problem: B={B}, H={H}, N={N}, D={D}")
    print(f"persistent cache hit (N={N}, D={D}): {cached}")

    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    sm_scale = D ** -0.5

    # First call — autotune or cache hit
    t0 = time.perf_counter()
    o = fused_attention(q, k, v, sm_scale)
    torch.cuda.synchronize()
    label = "use cached config" if cached else "autotune all configs"
    print(f"first call ({label}): {time.perf_counter() - t0:.2f}s")

    # Correctness
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    rel_err = (o - o_ref).abs().max().item() / max(o_ref.abs().max().item(), 1e-9)
    print(f"rel err vs torch.SDPA:  {rel_err:.2e}")
    print(f"best config:            {fused_attention_kernel.best_config}")
    print()

    # Bench vs torch SDPA
    triton_ms = _bench_cuda_events(lambda: fused_attention(q, k, v, sm_scale), args.warmup, args.iters)
    sdpa_ms   = _bench_cuda_events(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
        args.warmup, args.iters)

    flop = _attn_flops(B, H, N, D)
    print(f"{'kernel':<28}  {'time(ms)':>10}  {'TF/s':>8}  {'%SDPA':>8}")
    print("-" * 60)
    print(f"{'torch SDPA (causal)':<28}  {sdpa_ms:>10.4f}  {flop/sdpa_ms/1e12*1e3:>8.1f}  {100.0:>7.1f}%")
    print(f"{'Triton (autotuned)':<28}  {triton_ms:>10.4f}  {flop/triton_ms/1e12*1e3:>8.1f}  {100*sdpa_ms/triton_ms:>7.1f}%")


if __name__ == "__main__":
    main()
