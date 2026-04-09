#!/usr/bin/env python3

"""
SGEMM Benchmark: cuBLAS baseline vs one selected implementation.

FP32 in/out, FP32 accumulator. cuBLAS uses cublasSgemm (no tensor cores).

Examples:
    python bench_sgemm.py
    python bench_sgemm.py --size 2048 --method cuda
    python bench_sgemm.py --m 1024 --n 512 --k 2048 --method cutedsl
"""

import argparse
import ctypes

import cutlass.cute as cute
import numpy as np

from bench_utils import (
    BENCH_REGISTRY,
    CUDA_VARIANTS,
    CUTEDSL_VARIANTS,
    CUTLASS_VARIANTS,
    compile_and_benchmark_gpu,
    gflops,
    gpu_free,
    gpu_time_ms,
    load_cublas,
    load_cuda_lib,
    load_cutlass_lib,
    to_gpu,
)


_cutedsl_compiled = {}


def bench_cublas(M, N, K, warmup=5, iters=20):
    """Benchmark cuBLAS SGEMM TN: D = A^T * B."""
    blas = load_cublas()

    blas.cublasCreate_v2.restype = ctypes.c_int
    blas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    blas.cublasDestroy_v2.restype = ctypes.c_int
    blas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    blas.cublasSgemm_v2.restype = ctypes.c_int
    blas.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
    ]

    # TN: A stored (K,M) col-major, B stored (K,N) col-major, C (M,N) col-major
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float32))
    C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float32))

    dA = to_gpu(A_h)
    dB = to_gpu(B_h)
    dC = to_gpu(C_h)

    handle = ctypes.c_void_p()
    blas.cublasCreate_v2(ctypes.byref(handle))

    alpha = np.array([1.0], dtype=np.float32)
    beta = np.array([0.0], dtype=np.float32)

    CUBLAS_OP_T = 1
    CUBLAS_OP_N = 0

    def run():
        blas.cublasSgemm_v2(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            alpha.ctypes.data,
            ctypes.c_void_p(dA), K,
            ctypes.c_void_p(dB), K,
            beta.ctypes.data,
            ctypes.c_void_p(dC), M,
        )

    ms = gpu_time_ms(run, warmup, iters)

    blas.cublasDestroy_v2(handle)
    gpu_free(dA)
    gpu_free(dB)
    gpu_free(dC)
    return ms


def bench_cuda(M, N, K, warmup=5, iters=20):
    lib = load_cuda_lib()

    results = []
    for variant_name, symbol_name in sorted(CUDA_VARIANTS.items()):
        name = f"cuda:{variant_name}"
        dA = dB = dC = None
        try:
            # TN: A stored (K,M) col-major, B stored (K,N) col-major
            A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
            B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float32))
            C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float32))

            dA = to_gpu(A_h)
            dB = to_gpu(B_h)
            dC = to_gpu(C_h)

            kernel = getattr(lib, symbol_name)

            def run():
                kernel(
                    M, N, K,
                    ctypes.c_float(1.0),
                    ctypes.c_void_p(dA),
                    ctypes.c_void_p(dB),
                    ctypes.c_float(0.0),
                    ctypes.c_void_p(dC),
                )

            ms = gpu_time_ms(run, warmup, iters)
            results.append((name, ms, None))
        except Exception as err:
            results.append((name, None, err))
        finally:
            if dA is not None:
                safe_gpu_free(dA)
            if dB is not None:
                safe_gpu_free(dB)
            if dC is not None:
                safe_gpu_free(dC)

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


def bench_cutlass(M, N, K, warmup=5, iters=20):
    lib = load_cutlass_lib()

    results = []
    for variant_name, symbol_name in sorted(CUTLASS_VARIANTS.items()):
        name = f"cutlass:{variant_name}"
        dA = dB = dC = None
        try:
            # TN: A stored (K,M) col-major, B stored (K,N) col-major
            A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
            B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float32))
            C_h = np.asfortranarray(np.zeros((M, N), dtype=np.float32))

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
            if dA is not None:
                safe_gpu_free(dA)
            if dB is not None:
                safe_gpu_free(dB)
            if dC is not None:
                safe_gpu_free(dC)

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


def bench_cutedsl(M, N, K, warmup=5, iters=20):
    import cupy as cp
    from cutlass.cute.runtime import from_dlpack

    # TN: A stored (K,M) col-major, B stored (K,N) col-major, C (M,N) col-major
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float32))

    A_d = cp.array(A_h, order="F")      # (K,M) on GPU
    B_d = cp.array(B_h, order="F")      # (K,N) on GPU
    C_d = cp.zeros((M, N), dtype=cp.float32, order="F")

    # CuTeDSL expects mA=(M,K), mB=(N,K), mC=(M,N) — transpose A and B
    A_t = from_dlpack(A_d.T, assumed_align=16)   # (M,K):(K,1)
    B_t = from_dlpack(B_d.T, assumed_align=16)   # (N,K):(K,1)
    C_t = from_dlpack(C_d, assumed_align=16)     # (M,N):(1,M)

    results = []
    for variant_name, (gemm_cls, kwargs) in sorted(CUTEDSL_VARIANTS.items()):
        name = f"cutedsl:{variant_name}"
        key = (variant_name, M, N, K)
        try:
            def compile_fn():
                if key not in _cutedsl_compiled:
                    _cutedsl_compiled[key] = cute.compile(gemm_cls(**kwargs), A_t, B_t, C_t)

                compiled = _cutedsl_compiled[key]
                return lambda: compiled(A_t, B_t, C_t)

            ms, _ = compile_and_benchmark_gpu(compile_fn, warmup=warmup, iters=iters)
            results.append((name, ms, None))
        except Exception as err:
            results.append((name, None, err))

    results.sort(key=lambda item: float("inf") if item[1] is None else item[1])
    return results


METHOD_RUNNERS = {
    "cuda": bench_cuda,
    "cutlass": bench_cutlass,
    "cutedsl": bench_cutedsl,
}


def safe_gpu_free(ptr):
    try:
        gpu_free(ptr)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="SGEMM Benchmark")
    parser.add_argument("--size", type=int, default=1024,
                        help="Square problem size used when --m/--n/--k are not provided")
    parser.add_argument("--m", type=int, default=None, help="Problem size M")
    parser.add_argument("--n", type=int, default=None, help="Problem size N")
    parser.add_argument("--k", type=int, default=None, help="Problem size K")
    parser.add_argument("--method", choices=["all", *sorted(BENCH_REGISTRY.keys())], default="all",
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


def resolve_method_runner(method):
    return METHOD_RUNNERS[method]


def print_result_row(name, M, N, K, ms):
    print(f"{name:<35} {ms:>8.3f} ms  {gflops(M, N, K, ms):>8.0f} GF/s")


def print_failure_row(name, err):
    print(f"{name:<35} FAILED    {err}")


def main():
    args = parse_args()
    M, N, K = resolve_problem_shape(args)

    methods = sorted(METHOD_RUNNERS.keys()) if args.method == "all" else [args.method]

    print("SGEMM Benchmark (FP32 in/out, FP32 accumulator)")
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
        runner = resolve_method_runner(method)
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
