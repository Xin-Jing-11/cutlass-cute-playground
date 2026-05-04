#!/usr/bin/env python3
"""Persistent-CTA Triton FP16 GEMM kernels with best configs hardcoded.

Two kernels:
  - matmul_persistent:     non-TMA persistent (any SM 80+)
  - matmul_tma_persistent: TMA-based persistent, optional warp_specialize
                           (warp_specialize requires SM 100+ Blackwell)

The bare @triton.jit kernels are exposed for re-wrapping by
gemm_persistent_autotune.py, which sweeps a wider config space and writes the
chosen config to a JSON sidecar.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# ---------------------------------------------------------------------------
# Tile-id swizzle for L2 reuse (same as gemm.py)
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


# ===========================================================================
# Non-TMA persistent kernel
# ===========================================================================
@triton.jit
def matmul_kernel_persistent_jit(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_tiles_m * num_tiles_n
    offset_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offset_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offset_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offset_am = tl.where(offset_am < M, offset_am, 0)
        offset_bn = tl.where(offset_bn < N, offset_bn, 0)
        offset_am = tl.max_contiguous(tl.multiple_of(offset_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offset_bn = tl.max_contiguous(tl.multiple_of(offset_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for ki in range(k_tiles):
            offset_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak
            b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=offset_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.)
            b = tl.load(b_ptrs, mask=offset_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.)
            acc = tl.dot(a, b, acc)

        offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offset_cm[:, None] + stride_cn * offset_cn[None, :]
        c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = acc.to(tl.float8e4nv)
        else:
            c = acc.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


_BEST_PERSISTENT = triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_warps=8, num_stages=3,
)
matmul_kernel_persistent = triton.autotune(
    configs=[_BEST_PERSISTENT], key=['M', 'N', 'K'],
)(matmul_kernel_persistent_jit)


def matmul_persistent(a, b):
    M, K = a.shape
    K, N = b.shape
    assert a.dtype == b.dtype == torch.float16
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = lambda meta: (
        min(NUM_SMS,
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])),
    )
    matmul_kernel_persistent[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
    )
    return c


# ===========================================================================
# TMA persistent kernel (with optional warp_specialize and epilogue_subtile)
# ===========================================================================
@triton.jit
def matmul_kernel_tma_persistent_jit(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True,
                            warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
        offset_am = pid_m * BLOCK_SIZE_M
        offset_bn = pid_n * BLOCK_SIZE_N

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for ki in range(k_tiles):
            offset_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offset_am, offset_k])
            b = b_desc.load([offset_bn, offset_k])
            acc = tl.dot(a, b.T, acc)

        offset_cm = pid_m * BLOCK_SIZE_M
        offset_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c_desc.store([offset_cm, offset_cn], acc0.to(dtype))
            c_desc.store([offset_cm, offset_cn + BLOCK_SIZE_N // 2], acc1.to(dtype))
        else:
            c_desc.store([offset_cm, offset_cn], acc.to(dtype))


# When autotune sweeps BLOCK_SIZE_*, the host-built TMA descriptors must follow.
# The pre_hook patches descriptor .block_shape before each timed run.
def _tma_set_block_size_hook(nargs):
    BM = nargs["BLOCK_SIZE_M"]
    BN = nargs["BLOCK_SIZE_N"]
    BK = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BM, BK]
    nargs["b_desc"].block_shape = [BN, BK]
    if nargs.get("EPILOGUE_SUBTILE"):
        nargs["c_desc"].block_shape = [BM, BN // 2]
    else:
        nargs["c_desc"].block_shape = [BM, BN]


_BEST_TMA_PERSISTENT = triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_warps=8, num_stages=3,
    pre_hook=_tma_set_block_size_hook,
)
matmul_kernel_tma_persistent = triton.autotune(
    configs=[_BEST_TMA_PERSISTENT], key=['M', 'N', 'K'],
)(matmul_kernel_tma_persistent_jit)


def matmul_tma_persistent(a, b, warp_specialize=False, epilogue_subtile=False):
    """TMA persistent matmul. Internally transposes B from KxN -> NxK (TN layout)."""
    M, K = a.shape
    K, N = b.shape
    assert a.dtype == b.dtype == torch.float16
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    b_TN = b.T.contiguous()  # K-contiguous on both operands

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Initial descriptor block shapes — pre_hook may overwrite per autotune trial
    BM0 = _BEST_TMA_PERSISTENT.kwargs['BLOCK_SIZE_M']
    BN0 = _BEST_TMA_PERSISTENT.kwargs['BLOCK_SIZE_N']
    BK0 = _BEST_TMA_PERSISTENT.kwargs['BLOCK_SIZE_K']
    a_desc = TensorDescriptor.from_tensor(a,    [BM0, BK0])
    b_desc = TensorDescriptor.from_tensor(b_TN, [BN0, BK0])
    c_desc = TensorDescriptor.from_tensor(
        c, [BM0, BN0 // 2] if epilogue_subtile else [BM0, BN0])

    # TMA needs scratch for in-kernel descriptor encoding
    triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device='cuda'))

    grid = lambda meta: (
        min(NUM_SMS,
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])),
    )
    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        FP8_OUTPUT=False,
        EPILOGUE_SUBTILE=epilogue_subtile,
        NUM_SMS=NUM_SMS,
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
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    import argparse
    p = argparse.ArgumentParser(description="Persistent Triton matmul bench (best configs)")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"problem shape: M={M}, N={N}, K={K}")

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c_ref = a @ b
    flop = 2 * M * N * K

    runners = {
        'cuBLAS (torch.matmul)':           lambda: torch.matmul(a, b),
        'persistent (no TMA)':             lambda: matmul_persistent(a, b),
        'tma_persistent':                  lambda: matmul_tma_persistent(a, b),
        'tma_persistent + epi_subtile':    lambda: matmul_tma_persistent(a, b, epilogue_subtile=True),
    }

    # Correctness
    print("correctness vs torch.matmul:")
    for name, fn in runners.items():
        if name.startswith('cuBLAS'):
            continue
        c = fn()
        rel = (c - c_ref).abs().max().item() / max(c_ref.abs().max().item(), 1e-9)
        print(f"  {name:<35} rel_err={rel:.2e}")
    print()

    # Bench
    print(f"{'kernel':<35}  {'time(ms)':>10}  {'GF/s':>9}  {'%cuBLAS':>9}")
    print("-" * 70)
    cublas_ms = _bench_cuda_events(runners['cuBLAS (torch.matmul)'], args.warmup, args.iters)
    print(f"{'cuBLAS (torch.matmul)':<35}  {cublas_ms:>10.4f}  {flop/cublas_ms/1e9*1e3:>9.0f}  {100.0:>8.1f}%")
    for name, fn in runners.items():
        if name.startswith('cuBLAS'):
            continue
        ms = _bench_cuda_events(fn, args.warmup, args.iters)
        print(f"{name:<35}  {ms:>10.4f}  {flop/ms/1e9*1e3:>9.0f}  {100*cublas_ms/ms:>8.1f}%")


if __name__ == "__main__":
    main()
