#!/usr/bin/env python3

"""
HGEMM Accuracy Check: verify registered variants against a cuBLAS reference.

FP16 in/out, FP32 accumulator. All methods use TN col-major layout:
  D = alpha * A^T * B + beta * C
  A stored (K,M) col-major, B stored (K,N) col-major, C stored (M,N) col-major.

Examples:
    python check_hgemm.py
    python check_hgemm.py --method cutlass --size 256
"""

import argparse
import ctypes
import os
import re

import numpy as np

from bench_utils import (
    from_gpu,
    gpu_free,
    gpu_sync,
    load_cuda_lib,
    load_cutlass_lib,
    setup_cublas,
    to_gpu,
)


# ---------------------------------------------------------------------------
# HGEMM variant discovery
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))


def _discover_hgemm_variants(instantiate_path, symbol_prefix):
    variants = {}
    seen = set()
    pat_single = re.compile(r"^\s*INSTANTIATE_HGEMM_([A-Z0-9_]+)\((\d+)\)\s*$")
    pat_multi = re.compile(
        r"^\s*INSTANTIATE_HGEMM_([A-Z0-9_]+)\((\d+(?:\s*,\s*\d+)+)\)\s*$"
    )
    try:
        with open(instantiate_path, "r", encoding="utf-8") as f:
            for line in f:
                m = pat_single.match(line)
                if m:
                    variant = m.group(1).lower()
                    block = m.group(2)
                    key = f"{variant}_{block}" if variant in seen else variant
                    seen.add(variant)
                    variants[key] = f"{symbol_prefix}{variant}_{block}"
                    continue
                m = pat_multi.match(line)
                if m:
                    variant = m.group(1).lower()
                    params = "x".join(p.strip() for p in m.group(2).split(","))
                    key = f"{variant}_{params}"
                    seen.add(variant)
                    variants[key] = f"{symbol_prefix}{variant}_{params}"
    except OSError:
        pass
    return variants


CUTLASS_VARIANTS = _discover_hgemm_variants(
    os.path.join(ROOT, "CUTLASS", "hgemm", "instantiate.cu"),
    "cutlass_hgemm_",
)
CUDA_VARIANTS = _discover_hgemm_variants(
    os.path.join(ROOT, "CUDA", "hgemm", "instantiate.cu"),
    "cuda_hgemm_",
)

METHODS = sorted({"cutlass", "cuda"})

def safe_gpu_free(ptr):
    try:
        gpu_free(ptr)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="HGEMM accuracy check")
    parser.add_argument("--size", type=int, default=128,
                        help="Square problem size used when --m/--n/--k are not provided")
    parser.add_argument("--m", type=int, default=None, help="Problem size M")
    parser.add_argument("--n", type=int, default=None, help="Problem size N")
    parser.add_argument("--k", type=int, default=None, help="Problem size K")
    parser.add_argument("--method", choices=["all", *METHODS], default="all",
                        help="Implementation family to verify (default: all)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Optional variant name to verify (for example: warptiling)")
    parser.add_argument("--atol", type=float, default=5e-2, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=5e-2, help="Relative tolerance")
    return parser.parse_args()


def resolve_problem_shape(args):
    size = args.size
    M = args.m if args.m is not None else size
    N = args.n if args.n is not None else size
    K = args.k if args.k is not None else size
    return M, N, K


# ---------------------------------------------------------------------------
# cuBLAS reference: TN col-major, D = alpha * A^T * B + beta * C
# ---------------------------------------------------------------------------
def cublas_reference(M, N, K):
    """Run cuBLAS HGEMM and return (A_h, B_h, C_h, D_ref).

    Matrices use TN col-major layout matching CUTLASS convention.
    D_ref is the cuBLAS FP16 output cast to FP32.
    """
    blas, handle = setup_cublas()

    np.random.seed(42)
    A = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
    B = np.asfortranarray(np.random.randn(K, N).astype(np.float32))
    C = np.asfortranarray(np.random.randn(M, N).astype(np.float32))

    A_h = np.asfortranarray(A.astype(np.float16))
    B_h = np.asfortranarray(B.astype(np.float16))
    C_h = np.asfortranarray(C.astype(np.float16))

    dA = dB = dC = None
    try:
        dA = to_gpu(A_h)
        dB = to_gpu(B_h)
        dC = to_gpu(C_h)

        alpha = np.array([1.0], dtype=np.float32)
        beta = np.array([1.0], dtype=np.float32)

        CUBLAS_OP_T = 1
        CUBLAS_OP_N = 0
        CUDA_R_16F = 2
        CUBLAS_COMPUTE_32F = 68
        CUBLAS_GEMM_DEFAULT = -1

        blas.cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            alpha.ctypes.data,
            ctypes.c_void_p(dA), CUDA_R_16F, K,
            ctypes.c_void_p(dB), CUDA_R_16F, K,
            beta.ctypes.data,
            ctypes.c_void_p(dC), CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
        )
        gpu_sync()

        D_ref = from_gpu(dC, (M, N), np.float16, order="F").astype(np.float32)
    finally:
        for ptr in (dA, dB, dC):
            if ptr is not None:
                safe_gpu_free(ptr)
        blas.cublasDestroy_v2(handle)

    return A_h, B_h, C, D_ref


# ---------------------------------------------------------------------------
# CUTLASS: TN col-major, C = alpha * A^T * B + beta * C (in-place)
# ---------------------------------------------------------------------------
def check_cutlass(M, N, K, atol, rtol, variant=None):
    lib = load_cutlass_lib()
    A_h, B_h, C, D_ref = cublas_reference(M, N, K)

    results = []
    for variant_name, symbol_name in sorted(CUTLASS_VARIANTS.items()):
        if variant is not None and variant_name != variant:
            continue
        name = f"cutlass:{variant_name}"
        dA = dB = dC = None
        try:
            C_h = np.asfortranarray(C.astype(np.float16))
            dA = to_gpu(A_h)
            dB = to_gpu(B_h)
            dC = to_gpu(C_h)

            kernel = getattr(lib, symbol_name)
            kernel(
                M, N, K,
                ctypes.c_float(1.0),
                ctypes.c_void_p(dA), K,
                ctypes.c_void_p(dB), K,
                ctypes.c_float(1.0),
                ctypes.c_void_p(dC), M,
            )
            gpu_sync()

            D_out = from_gpu(dC, (M, N), np.float16, order="F").astype(np.float32)
            abs_err = float(np.max(np.abs(D_out - D_ref)))
            rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
            passed = bool(np.allclose(D_out, D_ref, atol=atol, rtol=rtol))
            results.append((name, passed, abs_err, rel_err, None))
        except Exception as err:
            results.append((name, False, None, None, err))
        finally:
            for ptr in (dA, dB, dC):
                if ptr is not None:
                    safe_gpu_free(ptr)

    return results


# ---------------------------------------------------------------------------
# CUDA: same TN col-major calling convention as CUTLASS (in-place C)
# ---------------------------------------------------------------------------
def check_cuda(M, N, K, atol, rtol, variant=None):
    lib = load_cuda_lib()
    A_h, B_h, C, D_ref = cublas_reference(M, N, K)

    results = []
    for variant_name, symbol_name in sorted(CUDA_VARIANTS.items()):
        if variant is not None and variant_name != variant:
            continue
        name = f"cuda:{variant_name}"
        dA = dB = dC = None
        try:
            C_h = np.asfortranarray(C.astype(np.float16))
            dA = to_gpu(A_h)
            dB = to_gpu(B_h)
            dC = to_gpu(C_h)

            kernel = getattr(lib, symbol_name)
            kernel(
                M, N, K,
                ctypes.c_float(1.0),
                ctypes.c_void_p(dA), K,
                ctypes.c_void_p(dB), K,
                ctypes.c_float(1.0),
                ctypes.c_void_p(dC), M,
            )
            gpu_sync()

            D_out = from_gpu(dC, (M, N), np.float16, order="F").astype(np.float32)
            abs_err = float(np.max(np.abs(D_out - D_ref)))
            rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
            passed = bool(np.allclose(D_out, D_ref, atol=atol, rtol=rtol))
            results.append((name, passed, abs_err, rel_err, None))
        except Exception as err:
            results.append((name, False, None, None, err))
        finally:
            for ptr in (dA, dB, dC):
                if ptr is not None:
                    safe_gpu_free(ptr)

    return results


def resolve_checker(method):
    return globals().get(f"check_{method}")


def print_result(name, passed, abs_err, rel_err, err):
    if err is not None:
        print(f"{name:<20} ERROR   {err}")
        return
    status = "PASS" if passed else "FAIL"
    print(f"{name:<60} {status:<6} abs_err={abs_err:.3e} rel_err={rel_err:.3e}")


def main():
    args = parse_args()
    M, N, K = resolve_problem_shape(args)

    print("HGEMM Accuracy Check (FP16 in/out, FP32 accumulator, cuBLAS reference)")
    print(f"Problem: M={M}, N={N}, K={K}")
    print(f"Selected: method={args.method}")
    if args.variant is not None:
        print(f"Selected: variant={args.variant}")
    print(f"Tolerances: atol={args.atol:g}, rtol={args.rtol:g}")
    print("-" * 72)

    methods = METHODS if args.method == "all" else [args.method]

    total_results = 0
    for method in methods:
        if args.method == "all":
            print(f"[{method}]")
        checker = resolve_checker(method)
        if checker is None:
            print(f"No checker implemented for method '{method}'.")
            continue
        results = checker(M, N, K, args.atol, args.rtol, args.variant)
        if not results:
            if args.method == "all":
                print("No variants matched the selected filter.")
            continue
        total_results += len(results)
        for result in results:
            print_result(*result)

    if total_results == 0:
        print("No variants matched the selected filter.")


if __name__ == "__main__":
    main()
