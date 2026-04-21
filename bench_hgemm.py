#!/usr/bin/env python3

"""
HGEMM Benchmark: cuBLAS baseline vs one selected implementation.

FP16 in/out, FP32 accumulator. cuBLAS uses CUBLAS_COMPUTE_32F.

Examples:
    python bench_hgemm.py
    python bench_hgemm.py --size 2048 --method cutlass
    python bench_hgemm.py --m 1024 --n 512 --k 2048 --method cutedsl
"""

import argparse
import ctypes
import os
import re

import cutlass.cute as cute
import numpy as np

from bench_utils import (
    compile_and_benchmark_gpu,
    gflops,
    gpu_free,
    gpu_time_ms,
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

from CuTeDSL.hgemm.instantiate import VARIANTS as _HGEMM_DSL_VARIANTS
CUTEDSL_VARIANTS = _HGEMM_DSL_VARIANTS

METHOD_RUNNERS = {}   # populated below after runner definitions

_cutedsl_compiled = {}


def safe_gpu_free(ptr):
    try:
        gpu_free(ptr)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# cuBLAS HGEMM baseline (TN col-major, FP32 accumulator)
# ---------------------------------------------------------------------------
def bench_cublas(M, N, K, warmup=5, iters=20):
    blas, handle = setup_cublas()

    # TN: A stored (K,M) col-major, B stored (K,N) col-major, C (M,N) col-major
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))
    C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float16))

    dA = to_gpu(A_h)
    dB = to_gpu(B_h)
    dC = to_gpu(C_h)

    alpha = np.array([1.0], dtype=np.float32)
    beta = np.array([0.0], dtype=np.float32)

    CUBLAS_OP_T = 1
    CUBLAS_OP_N = 0
    CUDA_R_16F = 2
    CUBLAS_COMPUTE_32F = 68
    CUBLAS_GEMM_DEFAULT = -1

    def run():
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

    ms = gpu_time_ms(run, warmup, iters)

    blas.cublasDestroy_v2(handle)
    gpu_free(dA); gpu_free(dB); gpu_free(dC)
    return ms


# ---------------------------------------------------------------------------
# CUTLASS C++ HGEMM (TN col-major, in-place C)
# ---------------------------------------------------------------------------
def bench_cutlass(M, N, K, warmup=5, iters=20):
    lib = load_cutlass_lib()

    results = []
    for variant_name, symbol_name in sorted(CUTLASS_VARIANTS.items()):
        name = f"cutlass:{variant_name}"
        dA = dB = dC = None
        try:
            # TN: A stored (K,M) col-major, B stored (K,N) col-major
            A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
            B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))
            C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float16))

            dA = to_gpu(A_h)
            dB = to_gpu(B_h)
            dC = to_gpu(C_h)

            kernel = getattr(lib, symbol_name)

            def run():
                kernel(
                    M, N, K,
                    ctypes.c_float(1.0),
                    ctypes.c_void_p(dA), K,
                    ctypes.c_void_p(dB), K,
                    ctypes.c_float(0.0),
                    ctypes.c_void_p(dC), M,
                )

            ms = gpu_time_ms(run, warmup, iters)
            results.append((name, ms, None))
        except Exception as err:
            results.append((name, None, err))
        finally:
            for ptr in (dA, dB, dC):
                if ptr is not None:
                    safe_gpu_free(ptr)

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


# ---------------------------------------------------------------------------
# CUDA C++ HGEMM (TN col-major, in-place C — same calling convention as CUTLASS)
# ---------------------------------------------------------------------------
def bench_cuda(M, N, K, warmup=5, iters=20):
    lib = load_cuda_lib()

    results = []
    for variant_name, symbol_name in sorted(CUDA_VARIANTS.items()):
        name = f"cuda:{variant_name}"
        dA = dB = dC = None
        try:
            A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
            B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))
            C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float16))

            dA = to_gpu(A_h)
            dB = to_gpu(B_h)
            dC = to_gpu(C_h)

            kernel = getattr(lib, symbol_name)

            def run():
                kernel(
                    M, N, K,
                    ctypes.c_float(1.0),
                    ctypes.c_void_p(dA), K,
                    ctypes.c_void_p(dB), K,
                    ctypes.c_float(0.0),
                    ctypes.c_void_p(dC), M,
                )

            ms = gpu_time_ms(run, warmup, iters)
            results.append((name, ms, None))
        except Exception as err:
            results.append((name, None, err))
        finally:
            for ptr in (dA, dB, dC):
                if ptr is not None:
                    safe_gpu_free(ptr)

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


# ---------------------------------------------------------------------------
# CuTe DSL HGEMM
# ---------------------------------------------------------------------------
def bench_cutedsl(M, N, K, warmup=5, iters=20):
    import cupy as cp
    from cutlass.cute.runtime import from_dlpack

    # TN: A stored (K,M) col-major, B stored (K,N) col-major, C (M,N) col-major
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))

    A_d = cp.array(A_h, order="F")      # (K,M) on GPU
    B_d = cp.array(B_h, order="F")      # (K,N) on GPU
    C_d = cp.zeros((M, N), dtype=cp.float16, order="F")
    D_d = cp.zeros((M, N), dtype=cp.float16, order="F")

    # CuTeDSL expects mA=(M,K), mB=(N,K), mC=(M,N), mD=(M,N)
    A_t = from_dlpack(A_d.T, assumed_align=16)   # (M,K):(K,1)
    B_t = from_dlpack(B_d.T, assumed_align=16)   # (N,K):(K,1)
    C_t = from_dlpack(C_d, assumed_align=16)     # (M,N):(1,M)
    D_t = from_dlpack(D_d, assumed_align=16)     # (M,N):(1,M)

    results = []
    for variant_name, (gemm_cls, kwargs) in sorted(CUTEDSL_VARIANTS.items()):
        name = f"cutedsl:{variant_name}"
        key = (variant_name, M, N, K)
        try:
            # Detect 3-tensor (in-place C) vs 4-tensor (separate C+D) interface
            import inspect
            sig = inspect.signature(gemm_cls.__call__)
            # Count tensor params (cute.Tensor annotations before alpha)
            n_tensors = sum(1 for p in list(sig.parameters.values())[1:]  # skip self
                           if p.annotation is cute.Tensor)
            if n_tensors == 3:
                tensors = (A_t, B_t, C_t)
            else:
                tensors = (A_t, B_t, C_t, D_t)

            def compile_fn(ts=tensors):
                if key not in _cutedsl_compiled:
                    _cutedsl_compiled[key] = cute.compile(
                        gemm_cls(**kwargs), *ts
                    )
                compiled = _cutedsl_compiled[key]
                return lambda: compiled(*ts)

            ms, _ = compile_and_benchmark_gpu(compile_fn, warmup=warmup, iters=iters)
            results.append((name, ms, None))
        except Exception as err:
            results.append((name, None, err))

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


METHOD_RUNNERS.update({
    "cuda": bench_cuda,
    "cutlass": bench_cutlass,
    "cutedsl": bench_cutedsl,
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HGEMM Benchmark")
    parser.add_argument("--size", type=int, default=1024,
                        help="Square problem size used when --m/--n/--k are not provided")
    parser.add_argument("--m", type=int, default=None, help="Problem size M")
    parser.add_argument("--n", type=int, default=None, help="Problem size N")
    parser.add_argument("--k", type=int, default=None, help="Problem size K")
    parser.add_argument("--method", choices=["all", *sorted(METHOD_RUNNERS.keys())], default="all",
                        help="Implementation family to benchmark against cuBLAS (default: all)")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    return parser.parse_args()


def resolve_problem_shape(args):
    size = args.size
    M = args.m if args.m is not None else size
    N = args.n if args.n is not None else size
    K = args.k if args.k is not None else size
    return M, N, K


def print_result_row(name, M, N, K, ms):
    print(f"{name:<60} {ms:>8.3f} ms  {gflops(M, N, K, ms):>8.0f} GF/s")


def print_failure_row(name, err):
    print(f"{name:<60} FAILED    {err}")


def main():
    args = parse_args()
    M, N, K = resolve_problem_shape(args)

    methods = sorted(METHOD_RUNNERS.keys()) if args.method == "all" else [args.method]

    print("HGEMM Benchmark (FP16 in/out, FP32 accumulator)")
    print(f"Problem: M={M}, N={N}, K={K}")
    print(f"Selected: method={args.method}")
    print("-" * 52)

    try:
        cublas_ms = bench_cublas(M, N, K, warmup=args.warmup, iters=args.iters)
    except Exception as err:
        print_failure_row("cuBLAS", err)
        return

    all_results = []
    for method in methods:
        runner = METHOD_RUNNERS[method]
        all_results.extend(runner(M, N, K, warmup=args.warmup, iters=args.iters))

    all_results.sort(key=lambda item: float("inf") if item[1] is None else item[1])

    print_result_row("cuBLAS", M, N, K, cublas_ms)
    for name, ms, err in all_results:
        if err is None:
            print_result_row(name, M, N, K, ms)
        else:
            print_failure_row(name, err)


if __name__ == "__main__":
    main()
