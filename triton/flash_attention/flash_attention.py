#!/usr/bin/env python3
"""Causal Flash Attention with TMA, Triton.

Hardcodes a single best config (BM=128, BN=64) found via
flash_attention_autotune.py at seq_len=2048, D_MODEL=64. Re-run that script
to retune for a different shape / GPU.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# Kernel — bare @triton.jit so flash_attention_autotune.py can re-wrap it
# with a wider config sweep.
# ---------------------------------------------------------------------------
@triton.jit
def fused_attention_kernel_jit(
    M, batch, head,
    desc_q, desc_k, desc_v, desc_o,
    sm_scale, seq_len,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= D_MODEL)
    bm = tl.program_id(0)              # which Q-tile within a (batch, head)
    bh = tl.program_id(1)              # which (batch, head) in [0, batch*head)

    qo_offset = bh * seq_len + bm * BLOCK_M   # start row for Q and O
    kv_offset = bh * seq_len                  # start row for K and V (advances by start_n)

    offset_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)

    # online softmax state in registers for the whole K-loop
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504   # sm_scale/log(2): use exp2 instead of exp

    q = desc_q.load([qo_offset, 0])    # [BLOCK_M, D_MODEL]

    # causal: sweep K only up to the Q-tile's last diagonal column
    hi = (bm + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        k = desc_k.load([kv_offset + start_n, 0])  # [BLOCK_N, D_MODEL]
        qk = tl.dot(q, k.T)

        n_idx = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offset_m[:, None] >= n_idx[None, :]
        qk = tl.where(causal_mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = desc_v.load([kv_offset + start_n, 0])  # [BLOCK_N, D_MODEL]
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    desc_o.store([qo_offset, 0], acc.to(q.dtype))

    m_ptrs = M + bh * seq_len + offset_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))


# ---------------------------------------------------------------------------
# Pre-hook: keep TMA descriptor block_shape in sync with autotune-chosen
# BLOCK_M / BLOCK_N. Exported for flash_attention_autotune.py to use.
# ---------------------------------------------------------------------------
def _fa_set_block_size_hook(nargs):
    BM = nargs["BLOCK_M"]
    BN = nargs["BLOCK_N"]
    D  = nargs["D_MODEL"]
    nargs["desc_q"].block_shape = [BM, D]
    nargs["desc_k"].block_shape = [BN, D]
    nargs["desc_v"].block_shape = [BN, D]
    nargs["desc_o"].block_shape = [BM, D]


# ---------------------------------------------------------------------------
# Best config — found via flash_attention_autotune.py.
# Wrapped in single-element @triton.autotune so the call site is identical
# whether tuning or not.
# ---------------------------------------------------------------------------
_BEST_FA_CONFIG = triton.Config(
    {'BLOCK_M': 128, 'BLOCK_N': 64},
    num_warps=4, num_stages=3,
    pre_hook=_fa_set_block_size_hook,
)
fused_attention_kernel = triton.autotune(
    configs=[_BEST_FA_CONFIG], key=['seq_len', 'D_MODEL'],
)(fused_attention_kernel_jit)


# q, k, v, shape [batch, num_heads, seq_len, d_model]
def fused_attention(q, k, v, sm_scale):
    assert q.shape == k.shape == v.shape
    assert q.shape[-1] in {16, 32, 64, 128, 256}

    batch, head, seq_len, d_model = q.shape

    o = torch.empty_like(q)
    M = torch.empty((batch, head, seq_len), device=q.device, dtype=torch.float32)

    bhs = batch * head * seq_len

    # Initial block_shapes — pre_hook overwrites per autotune trial. We pick
    # the hardcoded best up front so descriptor encoding is correct on first
    # call (before any pre_hook runs).
    BM0 = _BEST_FA_CONFIG.kwargs['BLOCK_M']
    BN0 = _BEST_FA_CONFIG.kwargs['BLOCK_N']
    desc_q = TensorDescriptor(q, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BM0, d_model])
    desc_k = TensorDescriptor(k, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BN0, d_model])
    desc_v = TensorDescriptor(v, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BN0, d_model])
    desc_o = TensorDescriptor(o, shape=[bhs, d_model], strides=[d_model, 1], block_shape=[BM0, d_model])

    triton.set_allocator(
        lambda size, align, _: torch.empty(size, dtype=torch.int8, device="cuda"))

    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * head, 1)
    fused_attention_kernel[grid](
        M, batch, head,
        desc_q, desc_k, desc_v, desc_o, sm_scale, seq_len,
        D_MODEL=d_model,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    for B, H, N, D in [(1, 2, 128, 64), (2, 4, 1024, 64), (1, 1, 256, 128)]:
        q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        sm_scale = D ** -0.5

        o_mine = fused_attention(q, k, v, sm_scale)
        qk_ref = (q @ k.transpose(-1, -2)).float() * sm_scale
        mask = torch.tril(torch.ones(N, N, device=DEVICE, dtype=torch.bool))
        qk_ref = qk_ref.masked_fill(~mask, float("-inf"))
        p_ref = qk_ref.softmax(dim=-1).to(torch.float16)
        o_ref = (p_ref @ v).to(torch.float16)

        max_abs_err = (o_mine - o_ref).abs().max().item()
        max_rel_err = ((o_mine - o_ref).abs() / o_ref.abs().clamp(min=1e-3)).max().item()
        ok = torch.allclose(o_mine, o_ref, atol=1e-2, rtol=1e-2)
        print(f"B={B} H={H} N={N} D={D}  max_abs={max_abs_err:.4e}  max_rel={max_rel_err:.4e}  {'PASS' if ok else 'FAIL'}")
