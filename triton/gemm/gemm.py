#!/usr/bin/env python3
"""Triton FP16 GEMM with the autotune-best config hardcoded.

The chosen tile/warp/stage combo was found by running gemm_autotune.py on a
4096x4096x4096 problem on RTX 5080 (SM 120). Re-run gemm_autotune.py to find
the best for a different shape / GPU and update _BEST_CONFIG below.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# Tile-id swizzle: 1-D M-grouping for L2 reuse
# ---------------------------------------------------------------------------
@triton.jit
def _compute_pid(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_tiles_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_tiles_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ---------------------------------------------------------------------------
# Kernel body — bare @triton.jit so it can be re-wrapped by gemm_autotune.py
# with a wider config list. Here it's wrapped with a single best config.
#
# All the explicit optimizations used:
# 1. smem tiling to reduce GMEM read and improve memory throughput
# 2. CTA level swizzle to increase L2 cache hit rate
# 3. tensor cores via tl.dot 
# 4. CP_ASYNC asynchronized copy with multiple stages to hide latency 
# All the implicit optimizations done by triton: 
# 1. auto vectorized data copy 
# 2. register tiling to increase FMA per memory load
# 3. auto smem swizzling for bank-conflict-free MMA loads
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel_jit(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    pidm, pidn = _compute_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offset_m = (pidm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_n = (pidn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offset_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offset_k[:, None] < (K - k * BLOCK_SIZE_K), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float16)

    offset_cm = pidm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pidn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offset_cm[:, None] + stride_cn * offset_cn[None, :]
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ---------------------------------------------------------------------------
# Best config (RTX 5080, 4096^3 FP16, found via gemm_autotune.py).
# Wrapped through @triton.autotune with a single-element list so the call site
# stays identical to the autotune-driven version — no codepath difference.
# ---------------------------------------------------------------------------
_BEST_CONFIG = triton.Config(
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
    num_warps=4, num_stages=3,
)
matmul_kernel = triton.autotune(
    configs=[_BEST_CONFIG], key=['M', 'N', 'K'],
)(matmul_kernel_jit)


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ===========================================================================
# TMA variant (non-persistent). Uses cp.async.bulk.tensor for gmem→smem and
# stores via TensorDescriptor. Works on SM 90+ (Hopper) and SM 100/120
# (Blackwell). On RTX 5080 the win over the cp.async path is small without
# warp-specialization (see gemm_persistent.py for the persistent + warp-spec
# combo where TMA earns its keep).
# ===========================================================================
@triton.jit
def matmul_kernel_tma_jit(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    pidm, pidn = _compute_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offset_m = pidm * BLOCK_SIZE_M
    offset_n = pidn * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # warp_specialize on the K-loop: producer warps issue TMAs, consumers run tl.dot.
    # Requires SM 100+ (Blackwell) — RTX 5080 (SM 120) is supported.
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        offset_k = k * BLOCK_SIZE_K
        a = a_desc.load([offset_m, offset_k])     # [BM, BK]
        b = b_desc.load([offset_n, offset_k])     # [BN, BK] (b is stored as N x K)
        acc = tl.dot(a, b.T, acc)                  # b.T -> [BK, BN]

    c_desc.store([offset_m, offset_n], acc.to(tl.float16))


# pre_hook keeps TMA descriptor block_shape in sync if BLOCK_SIZE_* changes
# (only relevant when re-wrapping with a wider autotune sweep).
def _tma_set_block_size_hook(nargs):
    BM = nargs["BLOCK_SIZE_M"]
    BN = nargs["BLOCK_SIZE_N"]
    BK = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BM, BK]
    nargs["b_desc"].block_shape = [BN, BK]
    nargs["c_desc"].block_shape = [BM, BN]


_BEST_TMA_CONFIG = triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_warps=8, num_stages=3,
    pre_hook=_tma_set_block_size_hook,
)
# WARP_SPECIALIZE is part of the cache key so different values get separate
# autotune entries (Triton recompiles per constexpr value anyway).
matmul_kernel_tma = triton.autotune(
    configs=[_BEST_TMA_CONFIG], key=['M', 'N', 'K', 'WARP_SPECIALIZE'],
)(matmul_kernel_tma_jit)


def matmul_tma(a, b, warp_specialize=False):
    """TMA-based non-persistent matmul. Internally transposes B to NxK so that
    both operands are K-contiguous (fast MMA atom on tensor cores).

    warp_specialize=True splits the K-loop's warps into TMA producer + tl.dot
    consumer roles (SM 100+ / Blackwell only)."""
    M, K = a.shape
    K, N = b.shape
    assert a.dtype == b.dtype == torch.float16
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    b_TN = b.T.contiguous()                        # K-contiguous on both operands

    init = _BEST_TMA_CONFIG.kwargs
    BM0, BN0, BK0 = init['BLOCK_SIZE_M'], init['BLOCK_SIZE_N'], init['BLOCK_SIZE_K']
    a_desc = TensorDescriptor.from_tensor(a,    [BM0, BK0])
    b_desc = TensorDescriptor.from_tensor(b_TN, [BN0, BK0])
    c_desc = TensorDescriptor.from_tensor(c,    [BM0, BN0])

    # TMA needs scratch for in-kernel descriptor encoding
    triton.set_allocator(
        lambda size, align, _: torch.empty(size, dtype=torch.int8, device='cuda'))

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    matmul_kernel_tma[grid](
        a_desc, b_desc, c_desc, M, N, K,
        WARP_SPECIALIZE=warp_specialize,
    )
    return c


# ---------------------------------------------------------------------------
# Quick correctness + bench when run directly.
# ---------------------------------------------------------------------------
def _bench_cuda_events(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run Triton matmul (best config) and compare to cuBLAS")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"problem shape: M={M}, N={N}, K={K}")
    print(f"non-TMA config: {_BEST_CONFIG}")
    print(f"TMA config:     {_BEST_TMA_CONFIG}\n")

    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c_ref = a @ b

    runners = [
        ("Triton (cp.async)",   lambda: matmul(a, b)),
        ("Triton TMA",          lambda: matmul_tma(a, b, warp_specialize=False)),
        ("Triton TMA + WS",     lambda: matmul_tma(a, b, warp_specialize=True)),
    ]
    for name, fn in runners:
        c = fn()
        rel = (c - c_ref).abs().max().item() / max(c_ref.abs().max().item(), 1e-9)
        print(f"  {name:<22} rel_err vs torch.matmul: {rel:.2e}")
    print()

    cublas_ms = _bench_cuda_events(lambda: torch.matmul(a, b), args.warmup, args.iters)
    flop = 2 * M * N * K
    print(f"{'kernel':<22}  {'time(ms)':>10}  {'GF/s':>9}  {'%cuBLAS':>9}")
    print("-" * 56)
    print(f"{'cuBLAS (torch.matmul)':<22}  {cublas_ms:>10.4f}  {flop/cublas_ms/1e9*1e3:>9.0f}  {100.0:>8.1f}%")
    for name, fn in runners:
        ms = _bench_cuda_events(fn, args.warmup, args.iters)
        print(f"{name:<22}  {ms:>10.4f}  {flop/ms/1e9*1e3:>9.0f}  {100*cublas_ms/ms:>8.1f}%")


if __name__ == "__main__":
    main()
