#!/usr/bin/env python3
"""
Autotune CUTLASS HGEMM kernels.

Regenerates CUTLASS/hgemm/instantiate.cu with batches of candidate configs,
rebuilds libcutlass_kernels.so, and benchmarks each config. Restores the
original instantiate.cu on exit (or interrupt).

Supported families:
  - wmma           : warp-level mma.sync, scalar s2r (no swizzle bank-free guarantee)
  - wmma_ldmatrix  : warp-level mma.sync + ldmatrix s2r
  - multistage     : above + cp.async pipelined gmem→smem, NUM_STAGES smem buffers

Examples:
    python tools/autotune_hgemm.py --family multistage --size 4096
    python tools/autotune_hgemm.py --family all --size 2048
    python tools/autotune_hgemm.py --family wmma_ldmatrix --dry-run
"""

import argparse
import atexit
import ctypes
import itertools
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cuda.bindings.runtime as cudart

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench_utils import (
    gflops, gpu_free, gpu_time_ms, to_gpu,
    setup_cublas,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CUTLASS_HGEMM_INSTANTIATE = ROOT / "CUTLASS" / "hgemm" / "instantiate.cu"
CUTLASS_BUILD_SO          = ROOT / "CUTLASS" / "build" / "libcutlass_kernels.so"
CUTLASS_HGEMM_SO_DIR      = ROOT / "CUTLASS" / "build" / "autotune_so_hgemm"

# SM120 budget: 128KB dynamic smem, 48KB static smem
MAX_DYNAMIC_SMEM = 128 * 1024
MAX_STATIC_SMEM  = 48  * 1024


# ---------------------------------------------------------------------------
# Preconditions (mirror kernel static_asserts + implicit requirements)
# ---------------------------------------------------------------------------
def _hgemm_common_ok(BM, BN, BK):
    """Common HGEMM constraints for 2x2x1 warp tiling + 128-bit cp.async g2s."""
    # MMA_K = 16, so BK must be multiple of 16
    if BK % 16:
        return False
    # 2 warps M × 16 = 32 M per MMA step  → BM multiple of 32
    if BM % 32:
        return False
    # 2 warps N × 8 = 16 N per MMA step → BN multiple of 16
    if BN % 16:
        return False
    # g2s: 128 threads, VEC=8 halves → BK_VEC = BK/8
    BK_VEC = BK // 8
    if BK_VEC == 0 or 128 % BK_VEC:
        return False
    ThrM = 128 // BK_VEC
    # BM and BN must be divisible by ThrM so tiled_copy covers whole tile integrally
    if BM % ThrM or BN % ThrM:
        return False
    return True


def wmma_precondition(params):
    BM, BN, BK = params
    if not _hgemm_common_ok(BM, BN, BK):
        return False
    # Static smem: both sA and sB as half (2B/elem)
    smem = (BM + BN) * BK * 2
    if smem > MAX_STATIC_SMEM:
        return False
    return True


def multistage_precondition(params):
    BM, BN, BK, STAGES = params
    if not _hgemm_common_ok(BM, BN, BK):
        return False
    if STAGES < 2 or STAGES > 8:
        return False
    # Dynamic smem: STAGES × (sA + sB)
    smem = STAGES * (BM + BN) * BK * 2
    if smem > MAX_DYNAMIC_SMEM:
        return False
    return True


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------
BASE_GRID = {
    "BM":  [64, 128, 256],
    "BN":  [64, 128, 256],
    "BK":  [16, 32, 64],
}

MULTISTAGE_GRID = {
    **BASE_GRID,
    "STAGES": [2, 3, 4, 5, 6],
}

HGEMM_FAMILIES = {
    "wmma": {
        "header":       "hgemm_wmma.cuh",
        "func":         "hgemm_wmma",
        "params":       ["BM", "BN", "BK"],
        "grid":         BASE_GRID,
        "precondition": wmma_precondition,
    },
    "wmma_ldmatrix": {
        "header":       "hgemm_wmma_ldmatrix.cuh",
        "func":         "hgemm_wmma_ldmatrix",
        "params":       ["BM", "BN", "BK"],
        "grid":         BASE_GRID,
        "precondition": wmma_precondition,
    },
    "multistage": {
        "header":       "hgemm_multistage.cuh",
        "func":         "hgemm_multistage",
        "params":       ["BM", "BN", "BK", "STAGES"],
        "grid":         MULTISTAGE_GRID,
        "precondition": multistage_precondition,
    },
}

FAMILY_NAMES = list(HGEMM_FAMILIES.keys())


def generate_configs(family_name):
    fam = HGEMM_FAMILIES[family_name]
    keys = fam["params"]
    values = [fam["grid"][k] for k in keys]
    return [c for c in itertools.product(*values) if fam["precondition"](c)]


# ---------------------------------------------------------------------------
# Symbol naming + instantiate.cu emission
# ---------------------------------------------------------------------------
def hgemm_symbol(family_name, params):
    tag = "x".join(str(p) for p in params)
    return f"cutlass_hgemm_{family_name}_{tag}"


def hgemm_emit(family_name, params):
    fam = HGEMM_FAMILIES[family_name]
    tmpl = ", ".join(str(p) for p in params)
    sym  = hgemm_symbol(family_name, params)
    return (
        f'extern "C" void {sym}(\n'
        f'    int m, int n, int k, float alpha,\n'
        f'    const half* A, int ldA, const half* B, int ldB,\n'
        f'    float beta, half* C, int ldC) {{\n'
        f'    {fam["func"]}<{tmpl}>(\n'
        f'        m, n, k, alpha,\n'
        f'        reinterpret_cast<const cute::half_t*>(A), ldA,\n'
        f'        reinterpret_cast<const cute::half_t*>(B), ldB,\n'
        f'        beta,\n'
        f'        reinterpret_cast<cute::half_t*>(C), ldC);\n'
        f'}}\n\n'
    )


def write_instantiate(batch):
    """batch: list of (family_name, params) tuples."""
    parts = ["// Auto-generated by autotune_hgemm.py\n",
             "#include <cuda_fp16.h>\n"]
    headers = {HGEMM_FAMILIES[fn]["header"] for fn, _ in batch}
    for h in sorted(headers):
        parts.append(f'#include "{h}"\n')
    parts.append("\n")
    for family_name, params in batch:
        parts.append(hgemm_emit(family_name, params))
    CUTLASS_HGEMM_INSTANTIATE.write_text("".join(parts))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build():
    try:
        proc = subprocess.run(
            ["make", "build-cutlass"],
            cwd=str(ROOT),
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "build timeout"
    if proc.returncode != 0:
        return False, (proc.stderr + proc.stdout)[-3000:]
    return True, ""


def snapshot_so(tag: str) -> Path:
    CUTLASS_HGEMM_SO_DIR.mkdir(parents=True, exist_ok=True)
    dst = CUTLASS_HGEMM_SO_DIR / f"libkernels_{tag}.so"
    shutil.copy2(CUTLASS_BUILD_SO, dst)
    return dst


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def _check_launch(fn):
    cudart.cudaGetLastError()
    fn()
    cudart.cudaDeviceSynchronize()
    err = cudart.cudaGetLastError()
    code = int(err[0] if isinstance(err, tuple) else err)
    return code == 0


def bench_config(lib, sym, dA, dB, dC, M, N, K, warmup, iters):
    kernel = getattr(lib, sym, None)
    if kernel is None:
        return None
    kernel.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
    ]
    kernel.restype = None

    def run():
        kernel(M, N, K, ctypes.c_float(1.0),
               ctypes.c_void_p(dA), K,
               ctypes.c_void_p(dB), K,
               ctypes.c_float(0.0),
               ctypes.c_void_p(dC), M)

    # Pre-check: does the kernel launch at all?
    if not _check_launch(run):
        return None

    try:
        ms = gpu_time_ms(run, warmup=warmup, iters=iters)
    except Exception:
        return None

    # Post-check: was there an error during the timed run?
    cudart.cudaDeviceSynchronize()
    err = cudart.cudaGetLastError()
    code = int(err[0] if isinstance(err, tuple) else err)
    if code != 0:
        return None

    # Sanity: FP16 tensor cores on SM120 peak ≈ 300 TFLOPS. A kernel claiming
    # more than the theoretical bound is a silent launch failure that slipped past.
    # Use a generous 500 TFLOPS upper bound as sanity guard.
    total_flops = 2.0 * M * N * K
    min_ms_sane = total_flops / (500e12) * 1e3   # ms
    if ms < min_ms_sane:
        return None

    return ms


def cublas_time_ms(M, N, K, dA, dB, dC, warmup, iters):
    """cuBLAS HGEMM (TN, FP16 in/out, FP32 accumulator) baseline."""
    blas, handle = setup_cublas()
    alpha = np.array([1.0], dtype=np.float32)
    beta  = np.array([0.0], dtype=np.float32)
    CUBLAS_OP_T, CUBLAS_OP_N = 1, 0
    CUDA_R_16F, CUBLAS_COMPUTE_32F = 2, 68
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

    ms = gpu_time_ms(run, warmup=warmup, iters=iters)
    blas.cublasDestroy_v2(handle)
    return ms


# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------
_backup_path: Path = None


def backup():
    global _backup_path
    bak = CUTLASS_HGEMM_INSTANTIATE.with_suffix(
        CUTLASS_HGEMM_INSTANTIATE.suffix + ".autotune.bak")
    if not bak.exists():
        shutil.copy2(CUTLASS_HGEMM_INSTANTIATE, bak)
    _backup_path = bak


def restore():
    if _backup_path and _backup_path.exists():
        shutil.copy2(_backup_path, CUTLASS_HGEMM_INSTANTIATE)
        os.utime(CUTLASS_HGEMM_INSTANTIATE, None)
        print(f"[autotune] restored {CUTLASS_HGEMM_INSTANTIATE}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_family(family_name, M, N, K, batch_size, warmup, iters, verbose,
               dA, dB, dC):
    configs = generate_configs(family_name)
    print(f"\n[{family_name}] {len(configs)} candidates after precondition filter")
    if not configs:
        return []

    results = []
    n_batches = (len(configs) + batch_size - 1) // batch_size
    n_launch_fail = 0

    for bi in range(n_batches):
        batch = configs[bi * batch_size : (bi + 1) * batch_size]
        write_instantiate([(family_name, p) for p in batch])

        t0 = time.time()
        ok, err = build()
        dt = time.time() - t0

        if not ok:
            print(f"  batch {bi+1}/{n_batches} BUILD FAILED in {dt:.1f}s")
            if verbose:
                print("  " + err.replace("\n", "\n  ")[-1000:])
            continue

        so_path = snapshot_so(f"{family_name}_{bi}")
        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError as e:
            print(f"  batch {bi+1}/{n_batches}: dlopen failed: {e}")
            continue

        print(f"  batch {bi+1}/{n_batches}: built in {dt:.1f}s ({len(batch)} kernels)")

        for params in batch:
            sym = hgemm_symbol(family_name, params)
            ms = bench_config(lib, sym, dA, dB, dC, M, N, K, warmup, iters)
            if ms is None:
                n_launch_fail += 1
                if verbose:
                    print(f"    LAUNCH FAIL {sym}")
                continue

            gf = gflops(M, N, K, ms)
            result = {
                "family": family_name,
                "params": params,
                "ms":     ms,
                "gflops": gf,
            }
            results.append(result)

            if verbose:
                param_names = HGEMM_FAMILIES[family_name]["params"]
                pstr = ",".join(f"{n}={v}" for n, v in zip(param_names, params))
                print(f"    {pstr}: {ms:7.3f} ms  {gf:7.0f} GF/s")

    if n_launch_fail:
        print(f"  [{family_name}] {n_launch_fail} configs failed cudaGetLastError check")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _param_string(family_name, params):
    names = HGEMM_FAMILIES[family_name]["params"]
    return " ".join(f"{n}={v}" for n, v in zip(names, params))


def print_top(results, top_n, title, cublas_ms=None):
    if not results:
        return
    print(f"\n{'='*80}")
    print(f" TOP {min(top_n, len(results))} {title}")
    print(f"{'='*80}")
    hdr = f"  {'family':<16}  {'params':<40}  {'ms':>8}  {'GF/s':>7}"
    if cublas_ms is not None:
        hdr += f"  {'%cuBLAS':>7}"
    print(hdr)
    print(f"  {'-'*16}  {'-'*40}  {'-'*8}  {'-'*7}"
          + (f"  {'-'*7}" if cublas_ms is not None else ""))

    for r in sorted(results, key=lambda x: x["ms"])[:top_n]:
        pstr = _param_string(r["family"], r["params"])
        line = f"  {r['family']:<16}  {pstr:<40}  {r['ms']:>8.3f}  {r['gflops']:>7.0f}"
        if cublas_ms is not None:
            pct = 100 * cublas_ms / r["ms"]  # higher = faster = closer to/above cuBLAS
            line += f"  {pct:>6.1f}%"
        print(line)


def print_best_per_family(results, cublas_ms=None):
    if not results:
        return
    best = {}
    for r in results:
        f = r["family"]
        if f not in best or r["ms"] < best[f]["ms"]:
            best[f] = r

    print(f"\n{'='*80}")
    print(" BEST PER FAMILY")
    print(f"{'='*80}")
    hdr = f"  {'family':<16}  {'params':<40}  {'ms':>8}  {'GF/s':>7}"
    if cublas_ms is not None:
        hdr += f"  {'%cuBLAS':>7}"
    print(hdr)
    print(f"  {'-'*16}  {'-'*40}  {'-'*8}  {'-'*7}"
          + (f"  {'-'*7}" if cublas_ms is not None else ""))

    for family_name, r in sorted(best.items()):
        pstr = _param_string(family_name, r["params"])
        line = f"  {family_name:<16}  {pstr:<40}  {r['ms']:>8.3f}  {r['gflops']:>7.0f}"
        if cublas_ms is not None:
            pct = 100 * cublas_ms / r["ms"]
            line += f"  {pct:>6.1f}%"
        print(line)


def write_csv(results, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "params", "ms", "gflops"])
        for r in results:
            w.writerow([
                r["family"],
                "x".join(str(p) for p in r["params"]),
                f"{r['ms']:.4f}",
                f"{r['gflops']:.0f}",
            ])


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def dry_run(family_names):
    print("DRY RUN — HGEMM candidate counts after precondition filtering:\n")
    total = 0
    for fn in family_names:
        configs = generate_configs(fn)
        print(f"  {fn:<16}  {len(configs):>5} configs")
        total += len(configs)
    print(f"\n  TOTAL: {total} configs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Autotune CUTLASS HGEMM kernels")
    ap.add_argument("--family", default="all",
                    choices=["all"] + FAMILY_NAMES)
    ap.add_argument("--size", type=int, default=4096,
                    help="Square problem size M=N=K")
    ap.add_argument("--batch-size", type=int, default=12,
                    help="Configs per build batch")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters",  type=int, default=10)
    ap.add_argument("--top",    type=int, default=10)
    ap.add_argument("--csv",    default="autotune_hgemm.csv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-cublas", action="store_true",
                    help="Skip cuBLAS baseline")
    args = ap.parse_args()

    family_names = FAMILY_NAMES if args.family == "all" else [args.family]

    if args.dry_run:
        dry_run(family_names)
        return

    # Backup and register restore
    backup()
    atexit.register(restore)
    signal.signal(signal.SIGINT,  lambda s, f: sys.exit(130))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(143))

    M = N = K = args.size

    # TN layout: A stored (K,M), B stored (K,N), C stored (M,N) column-major
    A_h = np.asfortranarray(np.random.randn(K, M).astype(np.float16))
    B_h = np.asfortranarray(np.random.randn(K, N).astype(np.float16))
    C_h = np.zeros((M, N), dtype=np.float16, order="F")

    dA = to_gpu(A_h)
    dB = to_gpu(B_h)
    dC = to_gpu(C_h)

    # cuBLAS baseline
    cublas_ms = None
    if not args.no_cublas:
        try:
            cublas_ms = cublas_time_ms(M, N, K, dA, dB, dC, args.warmup, args.iters)
            print(f"[cuBLAS] {cublas_ms:.3f} ms  "
                  f"{gflops(M, N, K, cublas_ms):.0f} GF/s")
        except Exception as e:
            print(f"[cuBLAS] skipped: {e}")

    all_results = []
    try:
        for family_name in family_names:
            r = run_family(
                family_name, M, N, K,
                args.batch_size, args.warmup, args.iters, args.verbose,
                dA, dB, dC,
            )
            all_results.extend(r)
            print_top(r, args.top,
                      f"{family_name} @ {args.size}\u00b3",
                      cublas_ms=cublas_ms)
    finally:
        gpu_free(dA)
        gpu_free(dB)
        gpu_free(dC)

    if all_results:
        print_top(all_results, args.top,
                  f"OVERALL @ {args.size}\u00b3",
                  cublas_ms=cublas_ms)
        print_best_per_family(all_results, cublas_ms=cublas_ms)
        write_csv(all_results, args.csv)
        print(f"\n{len(all_results)} results \u2192 {args.csv}")

    # Restore and rebuild
    restore()
    print("[autotune] rebuilding cutlass from original instantiate.cu")
    subprocess.run(["make", "build-cutlass"],
                   cwd=str(ROOT), capture_output=True)


if __name__ == "__main__":
    main()
