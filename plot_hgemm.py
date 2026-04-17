#!/usr/bin/env python3
"""
Run bench_hgemm.py at a given size and produce a grouped-by-family bar chart.

Results are saved under ./bench_results/ by default.
Same visual style as plot_sgemm.py.

Usage:
    python plot_hgemm.py                       # size=4096
    python plot_hgemm.py --size 2048
    python plot_hgemm.py --log path/to.log     # reuse a saved bench log
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
    """Short family_label (abbreviated) + compact config tag.

    CUTLASS uses 'wmma_*' prefix, CuTeDSL uses 'mma_*' — collapse to same label.
    """
    if kernel.startswith("multistage_"):
        return f"ms  {_multistage_cfg(kernel[len('multistage_'):])}"
    if kernel.startswith("wmma_ldmatrix_"):
        return f"wmma+ld  {_wmma_cfg(kernel[len('wmma_ldmatrix_'):])}"
    if kernel.startswith("mma_ldmatrix_"):
        return f"wmma+ld  {_wmma_cfg(kernel[len('mma_ldmatrix_'):])}"
    if kernel.startswith("wmma_"):
        return f"wmma  {_wmma_cfg(kernel[len('wmma_'):])}"
    if kernel.startswith("mma_"):
        return f"wmma  {_wmma_cfg(kernel[len('mma_'):])}"
    return kernel


def _wmma_cfg(body):
    """BMxBNxBK → 'BM×BN×BK'."""
    parts = body.split("x")
    if len(parts) == 3:
        return f"{parts[0]}×{parts[1]}×{parts[2]}"
    return body.replace("x", "×")


def _multistage_cfg(body):
    """BMxBNxBKxSTAGES → 'BM×BN×BK, s=STAGES'."""
    parts = body.split("x")
    if len(parts) == 4:
        bm, bn, bk, s = parts
        return f"{bm}×{bn}×{bk}, s={s}"
    return body.replace("x", "×")


FAMILY_RANK = {"wmma": 0, "wmma+ld": 1, "ms": 2}


def _family_rank(fam_label):
    for key, rank in FAMILY_RANK.items():
        if fam_label == key or fam_label.startswith(key + "  "):
            return rank
    return 99


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
BACKEND_COLORS = {
    "cutlass": "#2ca02c",
    "cutedsl": "#ff7f0e",
}
BACKEND_TAG = {
    "cutlass": "CUTLASS",
    "cutedsl": "CuTeDSL",
}
BACKEND_ORDER = ["cutlass", "cutedsl"]


def build_plot(entries, cublas_gflops, size, out_path):
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

    fig, ax = plt.subplots(figsize=(12, max(5, 1.0 + 0.55 * len(rows))))
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
    ax.set_title(
        f"HGEMM  —  M=N=K={size}, FP16 in/out, FP32 acc, RTX 5080 (SM 120)\n"
        f"wmma=mma.sync, +ld=ldmatrix s2r, ms=multistage cp.async; "
        f"tiles show BM×BN×BK (, s=stages)",
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
    ap = argparse.ArgumentParser(
        description="Run bench_hgemm.py and plot results grouped by kernel family.")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--bench-script", default="bench_hgemm.py")
    ap.add_argument("--log", default=None,
                    help="skip benchmark; read bench output from this log file")
    ap.add_argument("--iters",  type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    out_path = args.output or os.path.join(RESULTS_DIR, f"bench_hgemm_{args.size}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if args.log:
        with open(args.log) as f:
            text = f.read()
    else:
        cmd = [sys.executable, args.bench_script,
               "--size", str(args.size),
               "--iters", str(args.iters),
               "--warmup", str(args.warmup)]
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
