#!/usr/bin/env python3
"""
Plot SGEMM benchmark results from CSV data in bench_results/.

Usage:
    python plot_sgemm.py                       # size=4096, out=bench_results/bench_sgemm_4096_H100.png
    python plot_sgemm.py --size 2048
    python plot_sgemm.py --gpu RTX5080
    python plot_sgemm.py --csv bench_results/bench_sgemm_4096x4096x4096.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def canonical(kernel):
    """Return family_label for grouping (short family name + compact config tag)."""
    if kernel.startswith("naive"):
        return "naive"
    if kernel.startswith("smem"):
        rest = kernel[len("smem"):].lstrip("_")
        return "smem" + (f"  ({rest})" if rest == "mc" else "")
    if kernel.startswith("tiling_vectorize_"):
        return f"tl+v  {_tile_cfg(kernel[len('tiling_vectorize_'):])}"
    if kernel.startswith("tiling_"):
        return f"tl  {_tile_cfg(kernel[len('tiling_'):])}"
    if kernel.startswith("warptiling_vectorize_"):
        return f"wt+v  {_warp_cfg(kernel[len('warptiling_vectorize_'):])}"
    if kernel.startswith("warptiling_"):
        return f"wt  {_warp_cfg(kernel[len('warptiling_'):])}"
    if kernel.startswith("double_buffering_"):
        return f"db  {_warp_cfg(kernel[len('double_buffering_'):])}"
    return kernel


def _tile_cfg(body):
    """tiling config: show only BM×BN×BK (TM×TN is always 8×8 in the pruned set)."""
    mc, body = _split_mc(body)
    parts = body.split("x")
    if len(parts) == 5:
        bm, bn, bk, *_ = parts
        s = f"{bm}×{bn}×{bk}"
    else:
        s = body.replace("x", "×")
    return (f"(mc) {s}" if mc else s)


def _warp_cfg(body):
    """warptiling config: show only BM×BN×BK (WM×WN, WI, TM×TN are redundant
    once we've pruned to one config per family)."""
    mc, body = _split_mc(body)
    parts = body.split("x")
    if len(parts) == 9:
        bm, bn, bk, *_ = parts
        s = f"{bm}×{bn}×{bk}"
    else:
        s = body.replace("x", "×")
    return (f"(mc) {s}" if mc else s)


def _split_mc(body):
    if body.startswith("mc_"):
        return True, body[3:]
    return False, body


FAMILY_RANK = {
    "naive":  0,
    "smem":   1,
    "tl":     2,
    "tl+v":   3,
    "wt":     4,
    "wt+v":   5,
    "db":     6,
}


def _family_rank(fam_label):
    for key, rank in FAMILY_RANK.items():
        if fam_label == key or fam_label.startswith(key + "  "):
            return rank
    return 99


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
BACKEND_COLORS = {
    "cuda":    "#1f77b4",
    "cutlass": "#2ca02c",
    "cutedsl": "#ff7f0e",
}
BACKEND_TAG = {
    "cuda":    "CUDA",
    "cutlass": "CUTLASS",
    "cutedsl": "CuTeDSL",
}
BACKEND_ORDER = ["cuda", "cutlass", "cutedsl"]


GPU_INFO = {
    "H100":    "H100 NVL (SM 90a)",
    "RTX5080": "RTX 5080 (SM 120)",
}


def build_plot(entries, cublas_gflops, size, out_path, gpu="H100"):
    data = defaultdict(list)
    for backend, kernel, gflops in entries:
        fam = canonical(kernel)
        data[fam].append((backend, gflops))

    family_order = sorted(data.keys(), key=_family_rank)

    rows = []
    for fam in family_order:
        fam_entries = sorted(
            data[fam],
            key=lambda x: BACKEND_ORDER.index(x[0]) if x[0] in BACKEND_ORDER else 99,
        )
        for be, g in fam_entries:
            rows.append((fam, be, g))

    fig, ax = plt.subplots(figsize=(12, max(5, 1.0 + 0.38 * len(rows))))
    bar_h = 0.75
    family_gap = 0.9

    max_val = max((g for _, _, g in rows), default=1)
    x_max = max(cublas_gflops or 0, max_val) * 1.15
    x_label_offset = x_max * 0.015
    label_x = -x_max * 0.05
    family_x = -x_max * 0.38

    y = 0.0
    current_family = None
    family_first_y = {}

    for fam, be, g in rows:
        if fam != current_family:
            if current_family is not None:
                y += family_gap
            current_family = fam
            family_first_y[fam] = y

        color = BACKEND_COLORS.get(be, "#888888")
        ax.barh(y, g, height=bar_h, color=color,
                edgecolor="white", linewidth=0.6, zorder=3)
        pct_str = ""
        if cublas_gflops:
            pct = 100.0 * g / cublas_gflops
            pct_str = f"   ({pct:.1f}%)"
        ax.text(g + x_label_offset, y, f"{g:,}{pct_str}",
                va="center", fontsize=9, color="#222", zorder=4)
        ax.text(label_x, y, BACKEND_TAG.get(be, be),
                va="center", ha="right",
                fontsize=9, color=color, fontweight="bold")
        y += 1.0

    # Family boxes
    family_row_counts = defaultdict(int)
    for fam, _, _ in rows:
        family_row_counts[fam] += 1
    for fam, first_y in family_first_y.items():
        n = family_row_counts[fam]
        y_mid = first_y + (n - 1) / 2
        ax.text(family_x, y_mid, fam,
                va="center", ha="left", fontsize=10, fontweight="bold",
                color="#111",
                bbox=dict(facecolor="#f0f0f0", edgecolor="#999",
                          boxstyle="round,pad=0.4"))

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
    gpu_label = GPU_INFO.get(gpu, gpu)
    ax.set_title(
        f"SGEMM  —  M=N=K={size}, FP32, {gpu_label}\n"
        f"tl=tiling, wt=warptiling, db=double_buf, +v=vectorized, (mc)=M-contiguous; "
        f"tiles show BM×BN×BK",
        fontsize=11, pad=14,
    )
    ax.set_xlim(family_x * 1.05, x_max)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)

    patches = [
        mpatches.Patch(color=BACKEND_COLORS[b], label=BACKEND_TAG[b])
        for b in BACKEND_ORDER if b in {be for _, be, _ in rows}
    ]
    ax.legend(handles=patches, loc="upper right",
              fontsize=10, framealpha=0.95,
              title="Backend", title_fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SGEMM benchmark results from CSV data in bench_results/.")
    parser.add_argument("--size", type=int, default=4096,
                        help="Square problem size (matches bench CSV filename, default: 4096)")
    parser.add_argument("--gpu", choices=["H100", "RTX5080"], default="H100",
                        help="GPU label for plot title and output filename (default: H100)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--csv", default=None,
                        help="Path to bench CSV file (default: bench_results/bench_sgemm_<size>x<size>x<size>.csv)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        RESULTS_DIR, f"bench_sgemm_{args.size}_{args.gpu}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    csv_path = args.csv or os.path.join(
        RESULTS_DIR, f"bench_sgemm_{args.size}x{args.size}x{args.size}.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run bench_sgemm.py first to generate benchmark data.", file=sys.stderr)
        sys.exit(1)

    entries = []
    cublas_gflops = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row["backend"]
            kernel = row["kernel"]
            gf = int(row["gflops"])
            if backend == "cublas":
                cublas_gflops = gf
            else:
                entries.append((backend, kernel, gf))

    if not entries:
        print("No benchmark entries found in CSV — aborting.", file=sys.stderr)
        sys.exit(1)
    build_plot(entries, cublas_gflops, args.size, out_path, gpu=args.gpu)


if __name__ == "__main__":
    main()
