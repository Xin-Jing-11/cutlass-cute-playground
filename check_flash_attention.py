#!/usr/bin/env python3

"""
Flash Attention Accuracy Check: verify registered variants against a manual
PyTorch softmax-attention reference.

FP16 inputs / outputs, FP32 reference computation. Tolerances tuned for FP16.

Reference: O = softmax(Q @ K^T / sqrt(d_model), dim=-1) @ V  (no mask, no causal).

Examples:
    python check_flash_attention.py
    python check_flash_attention.py --batch 1 --heads 2 --seq-len 128 --d-model 32
    python check_flash_attention.py --variant v1_8x32
"""

import argparse
import ctypes
import math

import torch

from bench_flash_attention import FLASH_ATTENTION_VARIANTS, _variant_matches_d_model
from bench_utils import load_cuda_lib, load_cutlass_lib


def parse_args():
    p = argparse.ArgumentParser(description="Flash Attention accuracy check")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--variant", type=str, default=None,
                   help="Optional variant name (for example: v1_8x32)")
    p.add_argument("--atol", type=float, default=5e-3, help="Absolute tolerance")
    p.add_argument("--rtol", type=float, default=5e-3, help="Relative tolerance")
    return p.parse_args()


def pytorch_reference(B, H, S, D):
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda").contiguous()
    K = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda").contiguous()
    V = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda").contiguous()

    # Reference computed in FP32 for a stable ground truth.
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    O_ref = torch.matmul(attn, V.float()).contiguous()
    return Q, K, V, O_ref


def check_cuda(B, H, S, D, atol, rtol, variant=None):
    cuda_lib = load_cuda_lib()
    try:
        cutlass_lib = load_cutlass_lib()
    except OSError:
        cutlass_lib = None
    Q, K, V, O_ref = pytorch_reference(B, H, S, D)

    results = []
    for name, symbol_name in sorted(FLASH_ATTENTION_VARIANTS.items()):
        if variant is not None and name != variant:
            continue
        if not _variant_matches_d_model(name, D):
            continue
        backend = name.split(":", 1)[0]
        lib = cuda_lib if backend == "cuda" else cutlass_lib
        if lib is None:
            results.append((name, False, None, None, RuntimeError(f"{backend} lib not built")))
            continue
        try:
            out = torch.empty(B, H, S, D, dtype=torch.float16, device="cuda")

            kernel = getattr(lib, symbol_name)
            kernel.restype = None
            kernel.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            kernel(
                B, H, S, D,
                ctypes.c_void_p(Q.data_ptr()),
                ctypes.c_void_p(K.data_ptr()),
                ctypes.c_void_p(V.data_ptr()),
                ctypes.c_void_p(0),
                ctypes.c_void_p(out.data_ptr()),
            )
            torch.cuda.synchronize()

            out_f = out.float()
            diff = (out_f - O_ref).abs()
            abs_err = float(diff.max().item())
            rel_err = float(abs_err / (O_ref.abs().max().item() + 1e-6))
            passed = bool(torch.allclose(out_f, O_ref, atol=atol, rtol=rtol))
            results.append((name, passed, abs_err, rel_err, None))
        except Exception as err:
            results.append((name, False, None, None, err))

    return results


def print_result(name, passed, abs_err, rel_err, err):
    if err is not None:
        print(f"{name:<40} ERROR   {err}")
        return
    status = "PASS" if passed else "FAIL"
    print(f"{name:<40} {status:<6} abs_err={abs_err:.3e} rel_err={rel_err:.3e}")


def main():
    args = parse_args()
    B, H, S, D = args.batch, args.heads, args.seq_len, args.d_model

    print("Flash Attention Accuracy Check (PyTorch softmax-attention reference)")
    print(f"Problem: batch={B}, heads={H}, seq_len={S}, d_model={D}")
    if args.variant is not None:
        print(f"Selected: variant={args.variant}")
    print(f"Tolerances: atol={args.atol:g}, rtol={args.rtol:g}")
    print("-" * 72)

    results = check_cuda(B, H, S, D, args.atol, args.rtol, args.variant)
    if not results:
        print("No variants matched the selected filter.")
        return
    for result in results:
        print_result(*result)


if __name__ == "__main__":
    main()
