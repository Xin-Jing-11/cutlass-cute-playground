"""Shared GPU helpers for benchmark scripts."""

import ctypes
import glob
import os
import re

import numpy as np
import cuda.bindings.runtime as cudart


def _variant_class_name(prefix, variant):
    return prefix + "".join(token.capitalize() for token in variant.split("_"))


def _discover_cpp_variants(instantiate_path, symbol_prefix):
    variants = {}
    pattern = re.compile(r"^\s*INSTANTIATE_SGEMM_([A-Z0-9_]+)\((\d+)\)\s*$")
    try:
        with open(instantiate_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.match(line)
                if not match:
                    continue
                variant = match.group(1).lower()
                block = match.group(2)
                variants[variant] = f"{symbol_prefix}{variant}_{block}"
    except OSError:
        pass
    return variants


def _discover_cutedsl_variants():
    variants = {}
    root = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(root, "CuTeDSL", "sgemm", "sgemm_*.py"))):
        module_basename = os.path.splitext(os.path.basename(path))[0]  # sgemm_xxx
        if not module_basename.startswith("sgemm_"):
            continue
        variant = module_basename[len("sgemm_"):]
        class_name = _variant_class_name("Sgemm", variant)
        module_name = f"CuTeDSL.sgemm.{module_basename}"
        variants[variant] = (module_name, class_name)
    return variants


ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_VARIANTS = _discover_cpp_variants(
    os.path.join(ROOT, "CUDA", "sgemm", "instantiate.cu"),
    "cuda_sgemm_",
)
CUTLASS_VARIANTS = _discover_cpp_variants(
    os.path.join(ROOT, "CUTLASS", "sgemm", "instantiate.cu"),
    "cutlass_sgemm_",
)
CUTEDSL_VARIANTS = _discover_cutedsl_variants()

BENCH_REGISTRY = {
    "cuda": CUDA_VARIANTS,
    "cutlass": CUTLASS_VARIANTS,
    "cutedsl": CUTEDSL_VARIANTS,
}


def _check(err):
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")


def gflops(M, N, K, time_ms):
    # 2*M*N*K for gemm (multiply + add per element)
    # 3*M*N for epilogue: D = alpha * acc + beta * C (2 muls + 1 add)
    return (2.0 * M * N * K + 3.0 * M * N) * 1e-9 / (time_ms * 1e-3)


# ---------------------------------------------------------------------------
# GPU memory helpers using cuda-python
# ---------------------------------------------------------------------------
def gpu_alloc(nbytes):
    err, ptr = cudart.cudaMalloc(nbytes)
    _check(err)
    return ptr


def gpu_free(ptr):
    _check(cudart.cudaFree(ptr))


def to_gpu(host_arr):
    """Copy numpy array to GPU, return device pointer."""
    nbytes = host_arr.nbytes
    ptr = gpu_alloc(nbytes)
    _check(cudart.cudaMemcpy(ptr, host_arr.ctypes.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    return ptr


def from_gpu(ptr, shape, dtype, order="C"):
    """Copy from GPU to numpy array."""
    arr = np.empty(shape, dtype=dtype, order=order)
    _check(cudart.cudaMemcpy(arr.ctypes.data, ptr, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    return arr


def gpu_sync():
    _check(cudart.cudaDeviceSynchronize())


def gpu_time_ms(fn, warmup=5, iters=20):
    """Time a GPU function using CUDA events."""
    for _ in range(warmup):
        fn()
    gpu_sync()

    err, start = cudart.cudaEventCreate()
    _check(err)
    err, stop = cudart.cudaEventCreate()
    _check(err)

    _check(cudart.cudaEventRecord(start, 0))
    for _ in range(iters):
        fn()
    _check(cudart.cudaEventRecord(stop, 0))
    _check(cudart.cudaEventSynchronize(stop))

    err, ms = cudart.cudaEventElapsedTime(start, stop)
    _check(err)
    _check(cudart.cudaEventDestroy(start))
    _check(cudart.cudaEventDestroy(stop))
    return ms / iters


def benchmark_gpu_runner(run, warmup=5, iters=20):
    """Benchmark a pre-built GPU runner and return steady-state runtime in ms."""
    return gpu_time_ms(run, warmup=warmup, iters=iters)


def compile_and_benchmark_gpu(compile_fn, warmup=5, iters=20):
    """
    Split one-time Python-side compilation/setup from steady-state GPU timing.

    `compile_fn` should perform any untimed JIT/build/setup work and return a
    zero-argument callable that launches the already-compiled GPU work.

    Returns:
        (run_ms, run)
            run_ms: average steady-state runtime in milliseconds
            run: the callable returned by `compile_fn`, for reuse by the caller
    """
    run = compile_fn()
    run_ms = benchmark_gpu_runner(run, warmup=warmup, iters=iters)
    return run_ms, run


# ---------------------------------------------------------------------------
# Library loaders
# ---------------------------------------------------------------------------
_cutlass_lib = None
_cuda_lib = None
_cublas = None


def load_cutlass_lib():
    global _cutlass_lib
    if _cutlass_lib is None:
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'CUTLASS', 'build', 'libcutlass_kernels.so')
        _cutlass_lib = ctypes.CDLL(so_path)
    return _cutlass_lib


def load_cuda_lib():
    global _cuda_lib
    if _cuda_lib is None:
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'CUDA', 'build', 'libcuda_kernels.so')
        _cuda_lib = ctypes.CDLL(so_path)
    return _cuda_lib


def load_cublas():
    global _cublas
    if _cublas is not None:
        return _cublas
    for name in ['libcublas.so', 'libcublas.so.12', 'libcublas.so.11']:
        try:
            _cublas = ctypes.CDLL(name)
            return _cublas
        except OSError:
            continue
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    _cublas = ctypes.CDLL(f'{cuda_home}/lib64/libcublas.so')
    return _cublas


def setup_cublas():
    """Load cuBLAS and set up function signatures. Returns (blas, handle)."""
    blas = load_cublas()

    blas.cublasCreate_v2.restype = ctypes.c_int
    blas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    blas.cublasDestroy_v2.restype = ctypes.c_int
    blas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    blas.cublasGemmEx.restype = ctypes.c_int
    blas.cublasGemmEx.argtypes = [
        ctypes.c_void_p,                          # handle
        ctypes.c_int, ctypes.c_int,               # transa, transb
        ctypes.c_int, ctypes.c_int, ctypes.c_int, # m, n, k
        ctypes.c_void_p,                           # alpha
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # A, Atype, lda
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # B, Btype, ldb
        ctypes.c_void_p,                           # beta
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # C, Ctype, ldc
        ctypes.c_int,                              # computeType
        ctypes.c_int,                              # algo
    ]

    handle = ctypes.c_void_p()
    blas.cublasCreate_v2(ctypes.byref(handle))
    return blas, handle
