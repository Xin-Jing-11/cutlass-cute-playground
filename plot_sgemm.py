#!/usr/bin/env python3
"""
Run bench_sgemm.py at a given size and produce a grouped-by-family bar chart.

Results are saved under ./bench_results/ by default.

Usage:
    python plot_sgemm.py                       # size=4096, out=bench_results/bench_sgemm_4096.png
    python plot_sgemm.py --size 2048
    python plot_sgemm.py --size 8192 --output my_bench.png
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_bench_output(text):
    """Parse bench_sgemm.py stdout into (entries, cublas_gflops).

    entries: list of (backend, kernel, gflops)
    """
    entries = []
    cublas_gflops = None
    line_re = re.compile(r"^([A-Za-z0-9_:]+)\s+([\d\.]+)\s*ms\s+(\d+)\s*GF/s")

    for line in text.splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        name, _, gflops = m.group(1), float(m.group(2)), int(m.group(3))
        if name == "cuBLAS":
            cublas_gflops = gflops
            continue
        if ":" not in name:
            continue
        backend, kernel = name.split(":", 1)
        entries.append((backend, kernel, gflops))
    return entries, cublas_gflops


def canonical(kernel):
    """Return family_label for grouping."""
    if kernel.startswith("naive"):
        return "naive"
    if kernel.startswith("smem"):
        return "smem"
    if kernel.startswith("tiling_vectorize_"):
        return f"tiling_vec {kernel.replace('tiling_vectorize_', '')}"
    if kernel.startswith("tiling_"):
        return f"tiling {kernel.replace('tiling_', '')}"
    if kernel.startswith("warptiling_vectorize_"):
        return f"warptiling_vec {kernel.replace('warptiling_vectorize_', '')}"
    if kernel.startswith("warptiling_"):
        return f"warptiling {kernel.replace('warptiling_', '')}"
    if kernel.startswith("double_buffering_"):
        return f"double_buffering {kernel.replace('double_buffering_', '')}"
    return kernel


def family_display_name(fam):
    """Pretty-print a family label."""
    if fam in ("naive", "smem"):
        return fam
    parts = fam.split(" ", 1)
    if len(parts) != 2:
        return fam
    prefix, sizes = parts
    # tiling kernels: BMxBNxBKxTMxTN
    if prefix in ("tiling", "tiling_vec"):
        try:
            bm, bn, bk, tm, tn = sizes.split("x")
            return f"{prefix.replace('_vec', '+vec')}  {bm}×{bn}×{bk} / {tm}×{tn}"
        except ValueError:
            return fam
    # warptiling / double_buffering kernels: BMxBNxBKxWMxWNxWMITERxWNITERxTMxTN
    if prefix in ("warptiling", "warptiling_vec", "double_buffering"):
        try:
            bm, bn, bk, wm, wn, wmi, wni, tm, tn = sizes.split("x")
            return (f"{prefix.replace('_vec', '+vec')}  "
                    f"{bm}×{bn}×{bk}, {wm}×{wn}, WI={wmi}×{wni}, {tm}×{tn}")
        except ValueError:
            return fam
    return fam


def build_plot(entries, cublas_gflops, size, out_path):
    # Group by family
    data = defaultdict(list)
    for backend, kernel, gflops in entries:
        fam = canonical(kernel)
        data[fam].append((backend, gflops))

    # Order families from simplest to most optimized
    preferred_order = [
        "naive", "smem",
        "tiling 64x64x8x8x8",
        "tiling 64x64x16x8x8",
        "tiling 128x128x16x8x8",
        "tiling_vec 64x64x16x8x8",
        "tiling_vec 128x128x16x8x8",
        "warptiling 128x128x16x64x64x1x4x8x4",
        "warptiling_vec 128x128x16x64x64x1x4x8x4",
        "double_buffering 128x128x8x64x64x2x2x8x4",
    ]
    family_order = [f for f in preferred_order if f in data]
    # Append any unrecognized families at the end
    for f in data:
        if f not in family_order:
            family_order.append(f)

    backends = ["cuda", "cutlass", "cutedsl"]
    backend_colors = {
        "cuda":    "#1f77b4",
        "cutlass": "#2ca02c",
        "cutedsl": "#ff7f0e",
    }
    backend_display = {
        "cuda":    "CUDA",
        "cutlass": "CUTLASS (CuTe C++)",
        "cutedsl": "CuTeDSL (Python)",
    }

    # Build row list (sorted within family by backend)
    rows = []
    for fam in family_order:
        fam_entries = sorted(
            data[fam],
            key=lambda x: backends.index(x[0]) if x[0] in backends else 99,
        )
        for be, g in fam_entries:
            rows.append((fam, be, g))

    fig, ax = plt.subplots(figsize=(15, max(7, 1.0 + 0.35 * len(rows))))
    bar_h = 0.75
    family_gap = 0.9

    # Compute x-axis extent
    max_val = max((g for _, _, g in rows), default=1)
    x_max = max(cublas_gflops or 0, max_val) * 1.15
    x_label_offset = x_max * 0.015
    label_x = -x_max * 0.05
    family_x = -x_max * 0.33

    y = 0.0
    current_family = None
    family_first_y = {}

    for fam, be, g in rows:
        if fam != current_family:
            if current_family is not None:
                y += family_gap
            current_family = fam
            family_first_y[fam] = y

        color = backend_colors.get(be, "#888888")
        ax.barh(y, g, height=bar_h, color=color,
                edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(g + x_label_offset, y, f"{g:,}", va="center",
                fontsize=9, color="#222", zorder=4)
        tag = backend_display.get(be, be)
        ax.text(label_x, y, tag, va="center", ha="right",
                fontsize=9, color=color, fontweight="bold")
        y += 1.0

    # Family labels on the left
    family_row_counts = defaultdict(int)
    for fam, _, _ in rows:
        family_row_counts[fam] += 1
    for fam, first_y in family_first_y.items():
        n = family_row_counts[fam]
        y_mid = first_y + (n - 1) / 2
        ax.text(family_x, y_mid, family_display_name(fam),
                va="center", ha="left", fontsize=10, fontweight="bold",
                color="#111",
                bbox=dict(facecolor="#f0f0f0", edgecolor="#999",
                          boxstyle="round,pad=0.4"))

    # cuBLAS reference line
    if cublas_gflops is not None:
        ax.axvline(cublas_gflops, color="#d62728", linestyle="--",
                   linewidth=1.8, zorder=5)
        ax.text(cublas_gflops - x_label_offset, -0.9,
                f"cuBLAS  {cublas_gflops:,} GF/s",
                color="#d62728", ha="right", va="top",
                fontsize=10, fontweight="bold")

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Throughput (GFLOPS)", fontsize=12)
    ax.set_title(
        f"SGEMM Benchmark  —  M=N=K={size}, FP32 in/out, FP32 accumulator\n"
        f"NVIDIA RTX 5080 (SM 120 / Blackwell), CUDA 13.1",
        fontsize=13, pad=16,
    )
    ax.set_xlim(family_x * 1.05, x_max)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)

    # Legend
    backend_patches = [
        mpatches.Patch(color=backend_colors[b], label=backend_display[b])
        for b in backends if b in {be for _, be, _ in rows}
    ]
    ax.legend(handles=backend_patches,
              loc="lower right", fontsize=10, framealpha=0.95,
              title="Backend", title_fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results")


def main():
    parser = argparse.ArgumentParser(
        description="Run bench_sgemm.py and plot results grouped by kernel family.")
    parser.add_argument("--size", type=int, default=4096,
                        help="square problem size M=N=K (default: 4096)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help=f"output PNG path "
                             f"(default: {RESULTS_DIR}/bench_sgemm_<size>.png)")
    parser.add_argument("--bench-script", type=str, default="bench_sgemm.py",
                        help="path to bench_sgemm.py (default: bench_sgemm.py)")
    parser.add_argument("--log", type=str, default=None,
                        help="skip benchmark and read output from this log file")
    args = parser.parse_args()

    out_path = args.output or os.path.join(RESULTS_DIR, f"bench_sgemm_{args.size}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if args.log:
        with open(args.log) as f:
            text = f.read()
    else:
        cmd = [sys.executable, args.bench_script, "--size", str(args.size)]
        print(f"Running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        text = result.stdout
        print(text)

    entries, cublas_gflops = parse_bench_output(text)
    if not entries:
        print("No benchmark entries parsed — aborting.", file=sys.stderr)
        sys.exit(1)
    build_plot(entries, cublas_gflops, args.size, out_path)


if __name__ == "__main__":
    main()
