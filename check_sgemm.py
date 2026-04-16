#!/usr/bin/env python3

"""
SGEMM Accuracy Check: verify registered variants against a cuBLAS reference.

Examples:
    python check_sgemm.py
    python check_sgemm.py --method cuda
    python check_sgemm.py --method cutlass --size 256
    python check_sgemm.py --method cutedsl --m 128 --n 256 --k 64
"""

import argparse
import ctypes

import cutlass.cute as cute
import numpy as np

from bench_utils import (
    BENCH_REGISTRY,
    CUDA_VARIANTS,
    CUTLASS_VARIANTS,
    CUTEDSL_VARIANTS,
    from_gpu,
    gpu_free,
    gpu_sync,
    load_cuda_lib,
    load_cutlass_lib,
    setup_cublas,
    to_gpu,
)


_cutedsl_compiled = {}
METHODS = sorted(BENCH_REGISTRY.keys())


def safe_gpu_free(ptr):
    try:
        gpu_free(ptr)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="SGEMM accuracy check")
    parser.add_argument("--size", type=int, default=128,
                        help="Square problem size used when --m/--n/--k are not provided")
    parser.add_argument("--m", type=int, default=None, help="Problem size M")
    parser.add_argument("--n", type=int, default=None, help="Problem size N")
    parser.add_argument("--k", type=int, default=None, help="Problem size K")
    parser.add_argument("--method", choices=["all", *METHODS], default="all",
                        help="Implementation family to verify (default: all)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Optional variant name to verify (for example: naive)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    return parser.parse_args()


def resolve_problem_shape(args):
    size = args.size
    M = args.m if args.m is not None else size
    N = args.n if args.n is not None else size
    K = args.k if args.k is not None else size
    return M, N, K


def cublas_reference(M, N, K):
    """Run cuBLAS SGEMM and return (A, B, C, D_ref).

    TN col-major layout: D = alpha * A^T * B + beta * C.
    A stored (K,M) col-major, B stored (K,N) col-major, C stored (M,N) col-major.
    D_ref is the cuBLAS FP32 output.
    """
    blas, handle = setup_cublas()

    np.random.seed(42)
    A = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
    B = np.asfortranarray(np.random.randn(K, N).astype(np.float32))
    C = np.asfortranarray(np.random.randn(M, N).astype(np.float32))

    dA = dB = dC = None
    try:
        dA = to_gpu(A)
        dB = to_gpu(B)
        dC = to_gpu(C)

        alpha = np.array([1.0], dtype=np.float32)
        beta = np.array([1.0], dtype=np.float32)

        CUBLAS_OP_T = 1
        CUBLAS_OP_N = 0
        CUDA_R_32F = 0
        CUBLAS_COMPUTE_32F = 68
        CUBLAS_GEMM_DEFAULT = -1

        blas.cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            alpha.ctypes.data,
            ctypes.c_void_p(dA), CUDA_R_32F, K,
            ctypes.c_void_p(dB), CUDA_R_32F, K,
            beta.ctypes.data,
            ctypes.c_void_p(dC), CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
        )
        gpu_sync()

        D_ref = from_gpu(dC, (M, N), np.float32, order="F")
    finally:
        for ptr in (dA, dB, dC):
            if ptr is not None:
                safe_gpu_free(ptr)
        blas.cublasDestroy_v2(handle)

    return A, B, C, D_ref


def check_cuda(M, N, K, atol, rtol, variant=None):
    lib = load_cuda_lib()
    A, B, C, D_ref = cublas_reference(M, N, K)

    results = []
    for variant_name, symbol_name in sorted(CUDA_VARIANTS.items()):
        if variant is not None and variant_name != variant:
            continue
        name = f"cuda:{variant_name}"
        dA = dB = dC = None
        try:
            dA = to_gpu(A)
            dB = to_gpu(B)
            dC = to_gpu(C)

            kernel = getattr(lib, symbol_name)
            kernel(
                M, N, K,
                ctypes.c_float(1.0),
                ctypes.c_void_p(dA),
                ctypes.c_void_p(dB),
                ctypes.c_float(1.0),
                ctypes.c_void_p(dC),
            )
            gpu_sync()

            D_out = from_gpu(dC, (M, N), np.float32, order="F")
            abs_err = float(np.max(np.abs(D_out - D_ref)))
            rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
            passed = bool(np.allclose(D_out, D_ref, atol=atol, rtol=rtol))
            results.append((name, passed, abs_err, rel_err, None))
        except Exception as err:
            results.append((name, False, None, None, err))
        finally:
            if dA is not None:
                safe_gpu_free(dA)
            if dB is not None:
                safe_gpu_free(dB)
            if dC is not None:
                safe_gpu_free(dC)

    return results


def check_cutlass(M, N, K, atol, rtol, variant=None):
    lib = load_cutlass_lib()
    A, B, C, D_ref = cublas_reference(M, N, K)

    results = []
    for variant_name, symbol_name in sorted(CUTLASS_VARIANTS.items()):
        if variant is not None and variant_name != variant:
            continue
        name = f"cutlass:{variant_name}"
        dA = dB = dC = None
        try:
            dA = to_gpu(A)
            dB = to_gpu(B)
            dC = to_gpu(C)

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

            D_out = from_gpu(dC, (M, N), np.float32, order="F")
            abs_err = float(np.max(np.abs(D_out - D_ref)))
            rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
            passed = bool(np.allclose(D_out, D_ref, atol=atol, rtol=rtol))
            results.append((name, passed, abs_err, rel_err, None))
        except Exception as err:
            results.append((name, False, None, None, err))
        finally:
            if dA is not None:
                safe_gpu_free(dA)
            if dB is not None:
                safe_gpu_free(dB)
            if dC is not None:
                safe_gpu_free(dC)

    return results


def check_cutedsl(M, N, K, atol, rtol, variant=None):
    import cupy as cp
    from cutlass.cute.runtime import from_dlpack

    A, B, C, D_ref = cublas_reference(M, N, K)

    # A is (K,M) col-major, B is (K,N) col-major, C is (M,N) col-major
    A_d = cp.array(A, order="F")      # (K,M) on GPU
    B_d = cp.array(B, order="F")      # (K,N) on GPU
    C_d = cp.array(C, order="F")      # (M,N) on GPU

    # CuTeDSL kernel expects mA=(M,K), mB=(N,K), mC=(M,N) — transpose A and B
    A_t = from_dlpack(A_d.T, assumed_align=16)   # (M,K):(K,1)
    B_t = from_dlpack(B_d.T, assumed_align=16)   # (N,K):(K,1)
    C_t = from_dlpack(C_d, assumed_align=16)     # (M,N):(1,M)

    results = []
    for variant_name, (gemm_cls, kwargs) in sorted(CUTEDSL_VARIANTS.items()):
        if variant is not None and variant_name != variant:
            continue
        name = f"cutedsl:{variant_name}"
        key = (variant_name, M, N, K)
        try:
            if key not in _cutedsl_compiled:
                _cutedsl_compiled[key] = cute.compile(gemm_cls(**kwargs), A_t, B_t, C_t)

            compiled = _cutedsl_compiled[key]
            C_d.set(C)
            compiled(A_t, B_t, C_t, 1.0, 1.0)
            cp.cuda.runtime.deviceSynchronize()

            D_out = cp.asnumpy(C_d)
            abs_err = float(np.max(np.abs(D_out - D_ref)))
            rel_err = float(abs_err / (np.max(np.abs(D_ref)) + 1e-6))
            passed = bool(np.allclose(D_out, D_ref, atol=atol, rtol=rtol))
            results.append((name, passed, abs_err, rel_err, None))
        except Exception as err:
            results.append((name, False, None, None, err))

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

    print("SGEMM Accuracy Check (cuBLAS reference)")
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
