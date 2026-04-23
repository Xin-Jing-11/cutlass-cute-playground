#!/usr/bin/env python3

"""
Flash Attention Benchmark: PyTorch SDPA vs custom CUDA kernels.

FP16 storage, FP32 accumulators. Shapes: [batch, heads, seq_len, d_model], row-major.

Examples:
    python bench_flash_attention.py
    python bench_flash_attention.py --batch 2 --heads 8 --seq-len 2048 --d-model 64
"""

import argparse
import ctypes
import os
import re

import torch

from bench_utils import gpu_time_ms, load_cuda_lib, load_cutlass_lib


ROOT = os.path.dirname(os.path.abspath(__file__))


def _discover_flash_attention_variants(backend):
    path = os.path.join(ROOT, backend.upper(), "flash_attention", "instantiate.cu")
    pat = re.compile(
        r"^\s*INSTANTIATE_FLASH_ATTENTION_([A-Z0-9_]+)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$"
    )
    variants = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = pat.match(line)
                if m:
                    variant = m.group(1).lower()
                    bc, br = m.group(2), m.group(3)
                    name = f"{backend}:{variant}_{bc}x{br}"
                    symbol = f"{backend}_flash_attention_{variant}_{bc}x{br}"
                    variants[name] = symbol
    except OSError:
        pass
    return variants


FLASH_ATTENTION_VARIANTS = {
    **_discover_flash_attention_variants("cuda"),
    **_discover_flash_attention_variants("cutlass"),
}


def make_inputs(B, H, S, D):
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda").contiguous()
    K = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda").contiguous()
    V = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda").contiguous()
    return Q, K, V


def bench_sdpa(B, H, S, D, warmup=5, iters=20):
    Q, K, V = make_inputs(B, H, S, D)

    def run():
        torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

    return gpu_time_ms(run, warmup, iters)


_D_TAG = re.compile(r"_d(\d+)_")


def _variant_matches_d_model(variant_name, d_model):
    """Variants whose name contains `_d<N>_` only run when N matches d_model."""
    m = _D_TAG.search(variant_name)
    if m is None:
        return True
    return int(m.group(1)) == d_model


def bench_cuda(B, H, S, D, warmup=5, iters=20):
    cuda_lib = load_cuda_lib()
    try:
        cutlass_lib = load_cutlass_lib()
    except OSError:
        cutlass_lib = None

    Q, K, V = make_inputs(B, H, S, D)
    out = torch.empty(B, H, S, D, dtype=torch.float16, device="cuda")

    results = []
    for name, symbol in sorted(FLASH_ATTENTION_VARIANTS.items()):
        if not _variant_matches_d_model(name, D):
            continue
        backend = name.split(":", 1)[0]
        lib = cuda_lib if backend == "cuda" else cutlass_lib
        if lib is None:
            results.append((name, None, RuntimeError(f"{backend} library not built")))
            continue
        try:
            kernel = getattr(lib, symbol)
            kernel.restype = None
            kernel.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]

            def run():
                kernel(
                    B, H, S, D,
                    ctypes.c_void_p(Q.data_ptr()),
                    ctypes.c_void_p(K.data_ptr()),
                    ctypes.c_void_p(V.data_ptr()),
                    ctypes.c_void_p(0),
                    ctypes.c_void_p(out.data_ptr()),
                )

            ms = gpu_time_ms(run, warmup, iters)
            results.append((name, ms, None))
        except Exception as err:
            results.append((name, None, err))

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


def parse_args():
    # Default is the "golden" FA2/FA3 benchmark shape: B=16, H=16, S=4096, D=128.
    p = argparse.ArgumentParser(description="Flash Attention Benchmark")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def print_row(name, ms, ref_ms):
    speedup = ref_ms / ms if ms > 0 else float("inf")
    print(f"{name:<40} {ms:>10.3f} ms  {speedup:>6.2f}x")


def print_failure_row(name, err):
    print(f"{name:<40} FAILED    {err}")


def main():
    args = parse_args()
    B, H, S, D = args.batch, args.heads, args.seq_len, args.d_model

    print("Flash Attention Benchmark (FP16 storage, FP32 accumulators)")
    print(f"Problem: batch={B}, heads={H}, seq_len={S}, d_model={D}")
    print("-" * 60)

    sdpa_ms = bench_sdpa(B, H, S, D, warmup=args.warmup, iters=args.iters)
    print_row("sdpa (pytorch)", sdpa_ms, sdpa_ms)

    for name, ms, err in bench_cuda(B, H, S, D, warmup=args.warmup, iters=args.iters):
        if err is None:
            print_row(name, ms, sdpa_ms)
        else:
            print_failure_row(name, err)


if __name__ == "__main__":
    main()
