#!/usr/bin/env python3
"""FP8 GEMMs in Triton — both per-tensor scaled and MX (block-scaled).

Two kernels:
  - matmul_fp8        : per-tensor symmetric FP8 (E4M3), one scalar per operand
  - matmul_mxfp8      : block-scaled FP8 (E4M3 + E8M0 scales per 32 K-elements)

Both use FP32 accumulator and FP16 output. Requires SM 89+ for plain FP8,
SM 100+ for the MX path. Verified on RTX 5080 (SM 120).
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# Hardware-fixed for the MX path. Don't change.
MX_BLOCK_SIZE = 32

FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Tile-id swizzle for L2 reuse — shared by both kernels
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
# (1) Per-tensor scaled FP8
# ===========================================================================
@triton.jit
def matmul_fp8_kernel_jit(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    sa, sb,                                     # per-tensor dequant scales (FP32 scalars)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    # Dequant before downcast — raw acc can hit ~ 448*448*K, way past FP16's range.
    c = (acc * sa * sb).to(tl.float16)

    offset_cm = pidm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pidn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offset_cm[:, None] + stride_cn * offset_cn[None, :]
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_BEST_FP8_CONFIG = triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_warps=4, num_stages=3,
)
matmul_fp8_kernel = triton.autotune(
    configs=[_BEST_FP8_CONFIG], key=['M', 'N', 'K'],
)(matmul_fp8_kernel_jit)


def quantize_per_tensor_fp8(x):
    """Per-tensor symmetric FP8 quantization. Returns (x_fp8, scale)."""
    amax = x.abs().max().clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    x_scaled = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return x_scaled.to(torch.float8_e4m3fn), scale


def matmul_fp8(a_fp8, b_fp8, sa=1.0, sb=1.0):
    """A, B in FP8 E4M3. sa, sb: per-tensor dequant scales. Output: FP16."""
    M, K = a_fp8.shape
    K, N = b_fp8.shape
    assert a_fp8.dtype == torch.float8_e4m3fn
    assert b_fp8.dtype == torch.float8_e4m3fn
    c = torch.empty((M, N), device=a_fp8.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    matmul_fp8_kernel[grid](
        a_fp8, b_fp8, c,
        M, N, K,
        float(sa), float(sb),
        a_fp8.stride(0), a_fp8.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ===========================================================================
# (2) MX (block-scaled) FP8: E4M3 data + E8M0 scales per 32 K-elements
# ===========================================================================
def quantize_mxfp8(x):
    """High-precision tensor → (FP8 data, E8M0 scales [M, K/32] uint8)."""
    assert x.shape[-1] % MX_BLOCK_SIZE == 0, \
        f"K ({x.shape[-1]}) must be divisible by {MX_BLOCK_SIZE}"
    M, K = x.shape
    K_blocks = K // MX_BLOCK_SIZE

    blocks = x.float().reshape(M, K_blocks, MX_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1).clamp(min=2.0**-127)              # [M, K/32]

    s = torch.ceil(torch.log2(amax / FP8_E4M3_MAX)).clamp(-127, 128).to(torch.int32)
    scaled = blocks * (2.0 ** (-s.float())[..., None])
    x_fp8  = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    x_fp8  = x_fp8.reshape(M, K)
    x_scale_e8m0 = (s + 127).to(torch.uint8)
    return x_fp8, x_scale_e8m0


def dequantize_mxfp8(x_fp8, x_scale_e8m0):
    M, K = x_fp8.shape
    K_blocks = K // MX_BLOCK_SIZE
    s = x_scale_e8m0.to(torch.int32) - 127
    blocks = x_fp8.float().reshape(M, K_blocks, MX_BLOCK_SIZE)
    out = blocks * (2.0 ** s.float())[..., None]
    return out.reshape(M, K).to(torch.float16)


@triton.jit
def matmul_mxfp8_kernel_jit(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,                # B stored as [N, K] (TN)
    stride_cm, stride_cn,
    stride_a_sm, stride_a_sk,
    stride_b_sn, stride_b_sk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K % 32 == 0,
                     "BLOCK_SIZE_K must be a multiple of MX_BLOCK_SIZE (32)")
    SCALES_PER_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 32

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    pidm, pidn = _compute_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offset_m = (pidm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_n = (pidn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_n[:, None] * stride_bn + offset_k[None, :] * stride_bk

    offset_ks = tl.arange(0, SCALES_PER_BLOCK_K)
    a_scale_ptrs = a_scale_ptr + offset_m[:, None] * stride_a_sm + offset_ks[None, :] * stride_a_sk
    b_scale_ptrs = b_scale_ptr + offset_n[:, None] * stride_b_sn + offset_ks[None, :] * stride_b_sk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a       = tl.load(a_ptrs)
        b       = tl.load(b_ptrs)
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)
        # tl.dot_scaled does FP8×FP8 with implicit per-block dequant inside the tensor core.
        acc = tl.dot_scaled(a, a_scale, "e4m3", b.T, b_scale, "e4m3", acc=acc)
        a_ptrs       += BLOCK_SIZE_K * stride_ak
        b_ptrs       += BLOCK_SIZE_K * stride_bk
        a_scale_ptrs += SCALES_PER_BLOCK_K * stride_a_sk
        b_scale_ptrs += SCALES_PER_BLOCK_K * stride_b_sk

    c = acc.to(tl.float16)

    offset_cm = pidm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pidn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offset_cm[:, None] + stride_cn * offset_cn[None, :]
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_BEST_MXFP8_CONFIG = triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_warps=4, num_stages=3,
)
matmul_mxfp8_kernel = triton.autotune(
    configs=[_BEST_MXFP8_CONFIG], key=['M', 'N', 'K'],
)(matmul_mxfp8_kernel_jit)


def matmul_mxfp8(a_fp8, b_fp8_TN, a_scale, b_scale):
    """
    a_fp8        : FP8 E4M3,   [M, K]
    b_fp8_TN     : FP8 E4M3,   [N, K] (TN layout, both operands K-contiguous)
    a_scale      : E8M0 uint8, [M, K/32]
    b_scale      : E8M0 uint8, [N, K/32]
    Returns FP16, [M, N].
    """
    M, K = a_fp8.shape
    N, K2 = b_fp8_TN.shape
    assert K == K2 and K % MX_BLOCK_SIZE == 0
    assert a_scale.shape == (M, K // MX_BLOCK_SIZE)
    assert b_scale.shape == (N, K // MX_BLOCK_SIZE)

    c = torch.empty((M, N), device=a_fp8.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    matmul_mxfp8_kernel[grid](
        a_fp8, b_fp8_TN, c,
        a_scale, b_scale,
        M, N, K,
        a_fp8.stride(0),    a_fp8.stride(1),
        b_fp8_TN.stride(1), b_fp8_TN.stride(0),
        c.stride(0),        c.stride(1),
        a_scale.stride(0),  a_scale.stride(1),
        b_scale.stride(0),  b_scale.stride(1),
    )
    return c


# ===========================================================================
# Bench harness
# ===========================================================================
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
    p = argparse.ArgumentParser(description="Triton FP8 + MXFP8 GEMM — vs FP16 reference")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"problem shape: M={M}, N={N}, K={K}")
    print(f"plain FP8 config: {_BEST_FP8_CONFIG}")
    print(f"MXFP8 config:     {_BEST_MXFP8_CONFIG}\n")

    torch.manual_seed(0)
    a_f16 = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b_f16 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c_fp16 = torch.matmul(a_f16, b_f16)
    rms_ref = c_fp16.float().pow(2).mean().sqrt().item()

    # Plain FP8 (per-tensor)
    a_fp8_pt, sa = quantize_per_tensor_fp8(a_f16)
    b_fp8_pt, sb = quantize_per_tensor_fp8(b_f16)
    sa, sb = sa.item(), sb.item()
    c_fp8_pt = matmul_fp8(a_fp8_pt, b_fp8_pt, sa, sb)
    rms_pt = (c_fp8_pt.float() - c_fp16.float()).pow(2).mean().sqrt().item()

    # MXFP8 (block-scaled). B in [N, K] layout for fast MMA.
    a_fp8_mx, a_s = quantize_mxfp8(a_f16)
    b_fp8_mx, b_s = quantize_mxfp8(b_f16.T.contiguous())
    c_mx = matmul_mxfp8(a_fp8_mx, b_fp8_mx, a_s, b_s)
    rms_mx = (c_mx.float() - c_fp16.float()).pow(2).mean().sqrt().item()

    print("Correctness (RMS error / RMS ref):")
    print(f"  per-tensor FP8: {rms_pt/rms_ref:.4e}")
    print(f"  MXFP8:          {rms_mx/rms_ref:.4e}\n")

    fp16_ms = _bench_cuda_events(lambda: torch.matmul(a_f16, b_f16),                       args.warmup, args.iters)
    pt_ms   = _bench_cuda_events(lambda: matmul_fp8(a_fp8_pt, b_fp8_pt, sa, sb),           args.warmup, args.iters)
    mx_ms   = _bench_cuda_events(lambda: matmul_mxfp8(a_fp8_mx, b_fp8_mx, a_s, b_s),       args.warmup, args.iters)

    flop = 2 * M * N * K
    print(f"{'kernel':<28}  {'time(ms)':>10}  {'TF/s':>8}  {'%cuBLAS-FP16':>14}")
    print("-" * 68)
    print(f"{'cuBLAS FP16 (torch.matmul)':<28}  {fp16_ms:>10.4f}  {flop/fp16_ms/1e12*1e3:>8.1f}  {100.0:>13.1f}%")
    print(f"{'Triton FP8 (per-tensor)':<28}  {pt_ms:>10.4f}  {flop/pt_ms/1e12*1e3:>8.1f}  {100*fp16_ms/pt_ms:>13.1f}%")
    print(f"{'Triton MXFP8 (block, E4M3)':<28}  {mx_ms:>10.4f}  {flop/mx_ms/1e12*1e3:>8.1f}  {100*fp16_ms/mx_ms:>13.1f}%")


if __name__ == "__main__":
    main()
