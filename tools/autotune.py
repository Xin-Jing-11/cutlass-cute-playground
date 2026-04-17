#!/usr/bin/env python3
"""
Autotune CUDA and CUTLASS SGEMM kernels.

Regenerates instantiate.cu for each backend, rebuilds the .so, and
benchmarks every valid config. Restores instantiate.cu on exit.

Examples:
    python tools/autotune.py --backend cuda --kernel double_buffering --size 4096
    python tools/autotune.py --backend cutlass --kernel all --size 4096
    python tools/autotune.py --backend all --kernel all --size 4096
    python tools/autotune.py --backend all --kernel tiling --top 5

Output: top-N per family, overall top-N per backend, and a CSV.
"""

import argparse
import atexit
import ctypes
import itertools
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cuda.bindings.runtime as cudart

# Add project root to path so we can import bench_utils
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench_utils import gflops, gpu_free, gpu_time_ms, to_gpu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CUDA_INSTANTIATE    = ROOT / "CUDA"    / "sgemm" / "instantiate.cu"
CUTLASS_INSTANTIATE = ROOT / "CUTLASS" / "sgemm" / "instantiate.cu"

CUDA_BUILD_SO    = ROOT / "CUDA"    / "build" / "libcuda_kernels.so"
CUTLASS_BUILD_SO = ROOT / "CUTLASS" / "build" / "libcutlass_kernels.so"

CUDA_SO_DIR    = ROOT / "CUDA"    / "build" / "autotune_so"
CUTLASS_SO_DIR = ROOT / "CUTLASS" / "build" / "autotune_so"

# ---------------------------------------------------------------------------
# Precondition helpers (mirror kernel static_asserts + implicit requirements)
# ---------------------------------------------------------------------------
def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def tiling_precondition(params, vectorized=False):
    BM, BN, BK, TM, TN = params[:5]
    if not all(is_pow2(x) for x in (BM, BN, BK, TM, TN)):
        return False
    # static_assert: BM % TM == 0, BN % TN == 0
    if BM % TM or BN % TN:
        return False

    nthreads = (BM // TM) * (BN // TN)
    if nthreads < 32 or nthreads > 1024 or nthreads % 32:
        return False

    # smem: (BM*BK + BK*BN) floats must fit in 96KB
    if (BM * BK + BK * BN) * 4 > 96 * 1024:
        return False

    # g2s loading: each thread must load at least 1 element per tile
    # iterA = BK * TM * TN / BN,  iterB = BK * TM * TN / BM
    iterA = BK * TM * TN // BN
    iterB = BK * TM * TN // BM
    if iterA < 1 or iterB < 1:
        return False
    if (BK * TM * TN) % BN or (BK * TM * TN) % BM:
        return False

    # nrowsA = nthreads / BK — must be >= 1 (thread decomposition for g2s)
    if nthreads < BK:
        return False
    if nthreads % BK:
        return False

    if vectorized:
        if BK % 4:
            return False
        # iterA_vec = BK * TM * TN / BN / 4,  iterB_vec = BK * TM * TN / BM / 4
        iterA_v = BK * TM * TN // (BN * 4)
        iterB_v = BK * TM * TN // (BM * 4)
        if iterA_v < 1 or iterB_v < 1:
            return False
        # nrowsA = nthreads / (BK/4) — must be >= 1
        if nthreads < BK // 4:
            return False

    return True


def warptiling_precondition(params, vectorized=False, double_buffered=False):
    BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN = params[:9]
    if not all(is_pow2(x) for x in (BM, BN, BK, WM, WN, TM, TN)):
        return False
    if not (is_pow2(WMITER) and is_pow2(WNITER)):
        return False

    # static_assert: BM % WM == 0, BN % WN == 0
    if BM % WM or BN % WN:
        return False
    NWM, NWN = BM // WM, BN // WN
    nthreads = NWM * NWN * 32
    if nthreads < 32 or nthreads > 1024:
        return False

    # static_assert: WM % (WMITER * TM) == 0, WN % (WNITER * TN) == 0
    if WM % (WMITER * TM) or WN % (WNITER * TN):
        return False

    # Warp size constraint: NTM * NTN == 32
    WSUBM = WM // WMITER
    WSUBN = WN // WNITER
    NTM = WSUBM // TM
    NTN = WSUBN // TN
    if NTM * NTN != 32:
        return False

    # smem budget
    smem = (BM * BK + BK * BN) * 4
    if double_buffered:
        smem *= 2
    if smem > 96 * 1024:
        return False

    # g2s: each thread must load >= 1 element
    # scalar: iterA = BM * BK / NUM_THREADS >= 1
    if BM * BK < nthreads or BN * BK < nthreads:
        return False
    if (BM * BK) % nthreads or (BN * BK) % nthreads:
        return False

    # scalar g2s: nrowsA = NUM_THREADS / BK >= 1
    if nthreads < BK:
        return False
    if nthreads % BK:
        return False

    if vectorized or double_buffered:
        # B is always vectorized (float4)
        if BK % 4:
            return False
        BK_VEC = BK // 4
        # B: iterB = BN * BK / NUM_THREADS / 4 >= 1
        if BN * BK < 4 * nthreads:
            return False
        if (BN * BK) % (4 * nthreads):
            return False

    if vectorized:
        # A vectorized: iterA = BM * BK / NUM_THREADS / 4 >= 1
        if BM * BK < 4 * nthreads:
            return False
        if (BM * BK) % (4 * nthreads):
            return False

    # KC (MC=false) swizzle: Swizzle<SWZ_B-2, 2, SWZ_S> requires SWZ_B >= 3, i.e. BK >= 8
    if (vectorized or double_buffered) and BK < 8:
        return False

    # Register pressure estimate: skip configs with > 192 accumulators
    # (these will almost certainly spill badly or not launch)
    accum = TM * WMITER * TN * WNITER
    if accum > 192:
        return False

    return True


# ---------------------------------------------------------------------------
# Shared parameter grids
# ---------------------------------------------------------------------------
TILING_GRID = {
    "BM": [64, 128, 256],
    "BN": [64, 128, 256],
    "BK": [8, 16, 32],
    "TM": [4, 8, 16],
    "TN": [4, 8, 16],
}

WARPTILING_GRID = {
    "BM": [64, 128, 256],
    "BN": [64, 128, 256],
    "BK": [8, 16, 32],
    "WM": [32, 64, 128],
    "WN": [32, 64, 128],
    "WMITER": [1, 2, 4],
    "WNITER": [1, 2, 4],
    "TM": [4, 8],
    "TN": [4, 8],
}

DOUBLE_BUF_GRID = {
    "BM": [128, 256],
    "BN": [128, 256],
    "BK": [8, 16],
    "WM": [32, 64, 128],
    "WN": [32, 64, 128],
    "WMITER": [1, 2, 4],
    "WNITER": [1, 2, 4],
    "TM": [4, 8],
    "TN": [4, 8],
}

# ---------------------------------------------------------------------------
# Family definitions per backend
# ---------------------------------------------------------------------------
CUDA_FAMILIES = {
    "tiling": {
        "header": "sgemm_tiling.cuh",
        "func":   "sgemm_tiling",
        "params": ["BM", "BN", "BK", "TM", "TN"],
        "grid":   TILING_GRID,
        "precondition": lambda p: tiling_precondition(p),
        "mc_variants": True,
    },
    "tiling_vectorize": {
        "header": "sgemm_tiling_vectorize.cuh",
        "func":   "sgemm_tiling_vectorize",
        "params": ["BM", "BN", "BK", "TM", "TN"],
        "grid":   TILING_GRID,
        "precondition": lambda p: tiling_precondition(p, vectorized=True),
        "mc_variants": True,
    },
    "warptiling": {
        "header": "sgemm_warptiling.cuh",
        "func":   "sgemm_warptiling",
        "params": ["BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"],
        "grid":   WARPTILING_GRID,
        "precondition": lambda p: warptiling_precondition(p),
        "mc_variants": True,
    },
    "warptiling_vectorize": {
        "header": "sgemm_warptiling_vectorize.cuh",
        "func":   "sgemm_warptiling_vectorize",
        "params": ["BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"],
        "grid":   WARPTILING_GRID,
        "precondition": lambda p: warptiling_precondition(p, vectorized=True),
        "mc_variants": True,
    },
    "double_buffering": {
        "header": "sgemm_double_buffering.cuh",
        "func":   "sgemm_double_buffering",
        "params": ["BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"],
        "grid":   DOUBLE_BUF_GRID,
        "precondition": lambda p: warptiling_precondition(p, vectorized=True, double_buffered=True),
        "mc_variants": True,
    },
}

CUTLASS_FAMILIES = {
    "tiling": {
        "header": "sgemm_tiling.cuh",
        "func":   "sgemm_tiling",
        "device": "sgemm_tiling_device",
        "params": ["BM", "BN", "BK", "TM", "TN"],
        "grid":   TILING_GRID,
        "precondition": lambda p: tiling_precondition(p),
        "mc_variants": True,
    },
    "tiling_vectorize": {
        "header": "sgemm_tiling_vectorize.cuh",
        "func":   "sgemm_tiling_vectorize",
        "device": "sgemm_tiling_vectorize_device",
        "params": ["BM", "BN", "BK", "TM", "TN"],
        "grid":   TILING_GRID,
        "precondition": lambda p: tiling_precondition(p, vectorized=True),
        "mc_variants": True,
    },
    "warptiling": {
        "header": "sgemm_warptiling.cuh",
        "func":   "sgemm_warptiling",
        "device": "sgemm_warptiling_device",
        "params": ["BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"],
        "grid":   WARPTILING_GRID,
        "precondition": lambda p: warptiling_precondition(p),
        "mc_variants": True,
    },
    "warptiling_vectorize": {
        "header": "sgemm_warptiling_vectorize.cuh",
        "func":   "sgemm_warptiling_vectorize",
        "device": "sgemm_warptiling_vectorize_device",
        "params": ["BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"],
        "grid":   WARPTILING_GRID,
        "precondition": lambda p: warptiling_precondition(p, vectorized=True),
        "mc_variants": True,
    },
    "double_buffering": {
        "header": "sgemm_double_buffering.cuh",
        "func":   "sgemm_double_buffering",
        "device": "sgemm_double_buffering_device",
        "params": ["BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"],
        "grid":   DOUBLE_BUF_GRID,
        "precondition": lambda p: warptiling_precondition(p, vectorized=True, double_buffered=True),
        "mc_variants": False,
    },
}

# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def generate_configs(families, family_name):
    fam = families[family_name]
    keys = fam["params"]
    values = [fam["grid"][k] for k in keys]
    base = [c for c in itertools.product(*values) if fam["precondition"](c)]
    if fam.get("mc_variants"):
        return [(c, False) for c in base] + [(c, True) for c in base]
    return [(c, False) for c in base]


# ---------------------------------------------------------------------------
# CUDA: symbol naming + instantiate.cu emission
# ---------------------------------------------------------------------------
def cuda_symbol(family_name, params, mc: bool) -> str:
    tag = "x".join(str(p) for p in params)
    mc_str = "_mc" if mc else ""
    return f"cuda_sgemm_{family_name}{mc_str}_{tag}"


def cuda_emit(family_name, params, mc: bool) -> str:
    fam = CUDA_FAMILIES[family_name]
    tmpl = ", ".join(str(p) for p in params)
    if mc:
        tmpl += ", true"
    sym = cuda_symbol(family_name, params, mc)
    return (
        f'extern "C" void {sym}('
        f'int M, int N, int K, float alpha, '
        f'const float* A, const float* B, float beta, float* C) {{\n'
        f'    {fam["func"]}<{tmpl}>(M, N, K, alpha, A, B, beta, C);\n'
        f'}}\n\n'
    )


def cuda_write_instantiate(family_name, batch):
    fam = CUDA_FAMILIES[family_name]
    parts = [
        "// Auto-generated by autotune.py\n",
        f'#include "{fam["header"]}"\n\n',
    ]
    for params, mc in batch:
        parts.append(cuda_emit(family_name, params, mc))
    CUDA_INSTANTIATE.write_text("".join(parts))


# ---------------------------------------------------------------------------
# CUTLASS: symbol naming + instantiate.cu emission
# ---------------------------------------------------------------------------
def cutlass_symbol(family_name, params, mc: bool) -> str:
    tag = "x".join(str(p) for p in params)
    mc_str = "_mc" if mc else ""
    return f"cutlass_sgemm_{family_name}{mc_str}_{tag}"


def cutlass_emit(family_name, params, mc: bool) -> str:
    fam = CUTLASS_FAMILIES[family_name]
    tmpl = ", ".join(str(p) for p in params)
    if mc:
        tmpl += ", true"
    sym = cutlass_symbol(family_name, params, mc)
    return (
        f'extern "C" void {sym}('
        f'int m, int n, int k, float alpha, '
        f'const float* A, int ldA, const float* B, int ldB, '
        f'float beta, float* C, int ldC) {{\n'
        f'    {fam["func"]}<{tmpl}>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);\n'
        f'}}\n\n'
    )


def cutlass_write_instantiate(family_name, batch):
    fam = CUTLASS_FAMILIES[family_name]
    parts = [
        "// Auto-generated by autotune.py\n",
        f'#include "{fam["header"]}"\n\n',
    ]
    for params, mc in batch:
        parts.append(cutlass_emit(family_name, params, mc))
    CUTLASS_INSTANTIATE.write_text("".join(parts))


# ---------------------------------------------------------------------------
# Build + ptxas stat parsing
# ---------------------------------------------------------------------------
def build(target: str):
    """Run make build-{target}. Returns (ok, ptxas_stats, err_tail)."""
    try:
        proc = subprocess.run(
            ["make", f"build-{target}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, [], "build timeout"
    if proc.returncode != 0:
        return False, [], (proc.stderr + proc.stdout)[-3000:]

    combined = proc.stdout + proc.stderr
    stats = []
    blocks = re.split(r"ptxas info\s*:\s*Compiling entry function '", combined)
    for blk in blocks[1:]:
        m_name = re.match(r"([^']+)'", blk)
        if not m_name:
            continue
        mangled = m_name.group(1)
        m_spill = re.search(r"(\d+) bytes spill stores", blk)
        m_regs  = re.search(r"Used (\d+) registers", blk)
        m_smem  = re.search(r"(\d+) bytes smem", blk)
        stats.append({
            "mangled": mangled,
            "spill":   int(m_spill.group(1)) if m_spill else 0,
            "regs":    int(m_regs.group(1))  if m_regs  else 0,
            "smem":    int(m_smem.group(1))  if m_smem  else 0,
        })
    return True, stats, ""


def cuda_match_stats(stats, family_name, params):
    """Match ptxas stats by Li<N>E integer template encoding (CUDA kernels)."""
    fname = CUDA_FAMILIES[family_name]["func"]
    needle_parts = [f"Li{p}E" for p in params]
    for s in stats:
        if fname not in s["mangled"]:
            continue
        pos, ok = 0, True
        for part in needle_parts:
            idx = s["mangled"].find(part, pos)
            if idx < 0:
                ok = False
                break
            pos = idx + len(part)
        if ok:
            return s
    return None


def snapshot_so(so_dir: Path, build_so: Path, tag: str) -> Path:
    so_dir.mkdir(parents=True, exist_ok=True)
    dst = so_dir / f"libkernels_{tag}.so"
    shutil.copy2(build_so, dst)
    return dst


# ---------------------------------------------------------------------------
# Benchmark one config
# ---------------------------------------------------------------------------
def _cuda_check_launch(fn, M, N, K, dA, dB, dC):
    """Run kernel once and check cudaGetLastError. Returns True if OK."""
    # Drain any prior error
    cudart.cudaGetLastError()
    fn()
    cudart.cudaDeviceSynchronize()
    err = cudart.cudaGetLastError()
    code = int(err[0] if isinstance(err, tuple) else err)
    return code == 0


def bench_config_cuda(lib, sym, dA, dB, dC, M, N, K, warmup, iters):
    kernel = getattr(lib, sym, None)
    if kernel is None:
        return None
    kernel.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
    ]
    kernel.restype = None

    def run():
        kernel(M, N, K, ctypes.c_float(1.0),
               ctypes.c_void_p(dA), ctypes.c_void_p(dB),
               ctypes.c_float(0.0), ctypes.c_void_p(dC))

    if not _cuda_check_launch(run, M, N, K, dA, dB, dC):
        return None
    try:
        return gpu_time_ms(run, warmup=warmup, iters=iters)
    except Exception:
        return None


def bench_config_cutlass(lib, sym, dA, dB, dC, M, N, K, warmup, iters):
    kernel = getattr(lib, sym, None)
    if kernel is None:
        return None
    kernel.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,   # A, ldA
        ctypes.c_void_p, ctypes.c_int,   # B, ldB
        ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,   # C, ldC
    ]
    kernel.restype = None

    def run():
        kernel(M, N, K, ctypes.c_float(1.0),
               ctypes.c_void_p(dA), K,
               ctypes.c_void_p(dB), K,
               ctypes.c_float(0.0),
               ctypes.c_void_p(dC), M)

    if not _cuda_check_launch(run, M, N, K, dA, dB, dC):
        return None
    try:
        return gpu_time_ms(run, warmup=warmup, iters=iters)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backup / restore guards
# ---------------------------------------------------------------------------
_backups: dict[Path, Path] = {}


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".autotune.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    _backups[path] = bak


def restore(path: Path):
    bak = _backups.get(path)
    if bak and bak.exists():
        shutil.copy2(bak, path)
        os.utime(path, None)
        print(f"[autotune] restored {path}")


def restore_all():
    for path in list(_backups):
        restore(path)


# ---------------------------------------------------------------------------
# Run one family for one backend
# ---------------------------------------------------------------------------
def run_family(backend: str, family_name: str, M, N, K,
               batch_size, warmup, iters, max_spill, verbose,
               dA, dB, dC):
    if backend == "cuda":
        families   = CUDA_FAMILIES
        write_fn   = cuda_write_instantiate
        sym_fn     = cuda_symbol
        bench_fn   = bench_config_cuda
        build_tgt  = "cuda"
        build_so   = CUDA_BUILD_SO
        so_dir     = CUDA_SO_DIR
        inst_path  = CUDA_INSTANTIATE
    else:
        families   = CUTLASS_FAMILIES
        write_fn   = cutlass_write_instantiate
        sym_fn     = cutlass_symbol
        bench_fn   = bench_config_cutlass
        build_tgt  = "cutlass"
        build_so   = CUTLASS_BUILD_SO
        so_dir     = CUTLASS_SO_DIR
        inst_path  = CUTLASS_INSTANTIATE

    configs = generate_configs(families, family_name)
    tag = f"{backend}_{family_name}"
    print(f"\n[{tag}] {len(configs)} candidates after precondition filter")
    if not configs:
        return []

    results = []
    n_batches = (len(configs) + batch_size - 1) // batch_size
    n_launch_fail = 0

    for bi in range(n_batches):
        batch = configs[bi * batch_size : (bi + 1) * batch_size]
        write_fn(family_name, batch)

        t0 = time.time()
        ok, stats, err = build(build_tgt)
        dt = time.time() - t0

        if not ok:
            print(f"  batch {bi+1}/{n_batches} BUILD FAILED in {dt:.1f}s")
            if verbose:
                print("  " + err.replace("\n", "\n  ")[-1000:])
            continue

        so_path = snapshot_so(so_dir, build_so, f"{tag}_{bi}")
        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError as e:
            print(f"  batch {bi+1}/{n_batches}: dlopen failed: {e}")
            continue

        print(f"  batch {bi+1}/{n_batches}: built {len(stats)} kernels in {dt:.1f}s")

        for params, mc in batch:
            sym = sym_fn(family_name, params, mc)

            # ptxas stat lookup (CUDA only — CUTLASS types don't encode as Li<N>E)
            if backend == "cuda":
                st = cuda_match_stats(stats, family_name, params)
                if st is None:
                    if verbose:
                        print(f"    no ptxas stats for {sym}")
                    continue
                if st["spill"] > max_spill:
                    if verbose:
                        print(f"    skip {sym} (spill={st['spill']})")
                    continue
            else:
                st = {"regs": 0, "smem": 0, "spill": 0}

            ms = bench_fn(lib, sym, dA, dB, dC, M, N, K, warmup, iters)
            if ms is None:
                n_launch_fail += 1
                if verbose:
                    print(f"    LAUNCH FAIL {sym}")
                continue

            gf = gflops(M, N, K, ms)
            param_names = families[family_name]["params"]
            result = {
                "backend":    backend,
                "family":     family_name,
                "params":     params,
                "mc":         mc,
                "ms":         ms,
                "gflops":     gf,
                "regs":       st["regs"],
                "smem":       st["smem"],
                "spill":      st["spill"],
            }
            results.append(result)

            if verbose:
                pstr = ",".join(f"{n}={v}" for n, v in zip(param_names, params))
                mc_str = " MC" if mc else "   "
                print(f"    {mc_str} {pstr}: {ms:7.3f} ms  {gf:6.0f} GF/s  "
                      f"regs={st['regs']} spill={st['spill']}")

    if n_launch_fail:
        print(f"  [{tag}] {n_launch_fail} configs failed cudaGetLastError check")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_params(families, backend, family_name, params, mc):
    param_names = families[family_name]["params"]
    pstr = ",".join(f"{n}={v}" for n, v in zip(param_names, params))
    mc_str = " MC" if mc else "   "
    return f"{mc_str} {pstr}"


def print_top(all_results, top_n, title, families, backend):
    if not all_results:
        return
    fams = families
    print(f"\n{'='*80}")
    print(f" TOP {min(top_n, len(all_results))} {title}")
    print(f"{'='*80}")
    header = f"  {'family':<22} {'MC':>3}  {'params':<55}  {'ms':>8}  {'GF/s':>7}  {'regs':>5}  {'spill':>6}"
    print(header)
    print(f"  {'-'*22} {'-'*3}  {'-'*55}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*6}")
    for r in sorted(all_results, key=lambda x: x["ms"])[:top_n]:
        param_names = fams[r["family"]]["params"]
        pstr = " ".join(f"{n}={v}" for n, v in zip(param_names, r["params"]))
        mc_str = "Y" if r["mc"] else "N"
        spill_str = str(r["spill"]) if r["spill"] else "-"
        regs_str  = str(r["regs"])  if r["regs"]  else "-"
        print(f"  {r['family']:<22} {mc_str:>3}  {pstr:<55}  "
              f"{r['ms']:>8.3f}  {r['gflops']:>7.0f}  {regs_str:>5}  {spill_str:>6}")


def print_best_table(all_results, families):
    """Print a compact best-per-family table across all results."""
    if not all_results:
        return
    best: dict[tuple, dict] = {}
    for r in all_results:
        key = (r["backend"], r["family"])
        if key not in best or r["ms"] < best[key]["ms"]:
            best[key] = r

    print(f"\n{'='*80}")
    print(" BEST PARAMETERS PER METHOD")
    print(f"{'='*80}")
    header = f"  {'backend':<10} {'family':<22} {'MC':>3}  {'ms':>8}  {'GF/s':>7}  {'params'}"
    print(header)
    print(f"  {'-'*10} {'-'*22} {'-'*3}  {'-'*8}  {'-'*7}  {'-'*50}")

    for (backend, family_name), r in sorted(best.items()):
        param_names = families[family_name]["params"]
        pstr = " ".join(f"{n}={v}" for n, v in zip(param_names, r["params"]))
        mc_str = "Y" if r["mc"] else "N"
        print(f"  {backend:<10} {family_name:<22} {mc_str:>3}  "
              f"{r['ms']:>8.3f}  {r['gflops']:>7.0f}  {pstr}")


def write_csv(results, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "family", "mc", "params", "ms", "gflops",
                    "regs", "smem", "spill"])
        for r in results:
            w.writerow([
                r["backend"], r["family"],
                "mc" if r["mc"] else "kc",
                "x".join(str(p) for p in r["params"]),
                f"{r['ms']:.4f}", f"{r['gflops']:.0f}",
                r["regs"], r["smem"], r["spill"],
            ])


# ---------------------------------------------------------------------------
# Dry-run: show candidate counts without building
# ---------------------------------------------------------------------------
def dry_run(backends, family_names):
    print("DRY RUN — candidate counts after precondition filtering:\n")
    total = 0
    for backend in backends:
        families = CUDA_FAMILIES if backend == "cuda" else CUTLASS_FAMILIES
        for fn in family_names:
            if fn not in families:
                continue
            configs = generate_configs(families, fn)
            n_kc = sum(1 for _, mc in configs if not mc)
            n_mc = sum(1 for _, mc in configs if mc)
            print(f"  {backend:>8}:{fn:<22}  {len(configs):>5} total  "
                  f"({n_kc} KC + {n_mc} MC)")
            total += len(configs)
    print(f"\n  TOTAL: {total} configs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FAMILY_NAMES = sorted(set(list(CUDA_FAMILIES.keys()) + list(CUTLASS_FAMILIES.keys())))


def main():
    ap = argparse.ArgumentParser(
        description="Autotune CUDA and CUTLASS SGEMM kernels"
    )
    ap.add_argument("--backend", default="all",
                    choices=["all", "cuda", "cutlass"])
    ap.add_argument("--kernel", default="all",
                    choices=["all"] + FAMILY_NAMES)
    ap.add_argument("--size", type=int, default=2048,
                    help="Square problem size M=N=K (must be power of 2)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Configs per build batch (smaller = faster build, more batches)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters",  type=int, default=10)
    ap.add_argument("--max-spill", type=int, default=64,
                    help="(CUDA only) skip configs with more spill bytes than this")
    ap.add_argument("--top",    type=int, default=10)
    ap.add_argument("--csv",    default="autotune_results.csv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print candidate counts, don't build or benchmark")
    args = ap.parse_args()

    if not is_pow2(args.size):
        print(f"--size must be a power of 2 (got {args.size})")
        sys.exit(2)

    backends = ["cuda", "cutlass"] if args.backend == "all" else [args.backend]
    all_family_names = FAMILY_NAMES if args.kernel == "all" else [args.kernel]

    if args.dry_run:
        dry_run(backends, all_family_names)
        return

    # Backup instantiate files for backends we'll touch
    for b in backends:
        inst = CUDA_INSTANTIATE if b == "cuda" else CUTLASS_INSTANTIATE
        backup(inst)
    atexit.register(restore_all)
    signal.signal(signal.SIGINT,  lambda s, f: sys.exit(130))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(143))

    M = N = K = args.size

    A_cuda = np.asfortranarray(np.random.randn(K, M).astype(np.float32))
    B_cuda = np.asfortranarray(np.random.randn(K, N).astype(np.float32))
    C_cuda = np.zeros((M, N), dtype=np.float32, order="F")

    dA = to_gpu(A_cuda)
    dB = to_gpu(B_cuda)
    dC = to_gpu(C_cuda)

    all_results = []
    try:
        for backend in backends:
            families = CUDA_FAMILIES if backend == "cuda" else CUTLASS_FAMILIES
            family_names = [f for f in all_family_names if f in families]

            for family_name in family_names:
                r = run_family(
                    backend, family_name, M, N, K,
                    args.batch_size, args.warmup, args.iters,
                    args.max_spill, args.verbose,
                    dA, dB, dC,
                )
                all_results.extend(r)
                print_top(r, args.top,
                          f"{backend.upper()} {family_name} @ {args.size}\u00b3",
                          families, backend)
    finally:
        gpu_free(dA)
        gpu_free(dB)
        gpu_free(dC)

    if all_results:
        for backend in backends:
            br = [r for r in all_results if r["backend"] == backend]
            families = CUDA_FAMILIES if backend == "cuda" else CUTLASS_FAMILIES
            print_top(br, args.top,
                      f"{backend.upper()} OVERALL @ {args.size}\u00b3",
                      families, backend)

        print_best_table(all_results,
                         {**CUDA_FAMILIES, **CUTLASS_FAMILIES})
        write_csv(all_results, args.csv)
        print(f"\n{len(all_results)} results \u2192 {args.csv}")

    # Restore and rebuild both touched backends
    restore_all()
    for b in backends:
        print(f"[autotune] rebuilding {b} from original instantiate.cu")
        subprocess.run(["make", f"build-{b}"], cwd=str(ROOT), capture_output=True)


if __name__ == "__main__":
    main()
