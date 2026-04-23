# Flash Attention — Math Notes

## Standard attention

Given $Q, K, V \in \mathbb{R}^{N \times d}$:

$$
S = \frac{Q K^{\top}}{\sqrt{d}}, \quad
P = \operatorname{softmax}(S), \quad
O = P V
$$

Row-wise softmax uses the shift-invariant form

$$
m_i = \max_j S_{ij}, \qquad
\ell_i = \sum_j e^{S_{ij} - m_i}, \qquad
P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i}
$$

Materializing $S$ and $P$ costs $\mathcal{O}(N^2)$ HBM traffic. Flash attention avoids this by tiling and fusing softmax into the mainloop.

## Online-softmax recurrence

Split the keys/values into column blocks $j = 1, \dots, T_c$. For a fixed query row $i$, let $S^{(j)}$ be the score tile for block $j$, with tile statistics

$$
\tilde m^{(j)} = \max_{k \in j} S_{ik}, \qquad
\tilde\ell^{(j)} = \sum_{k \in j} e^{S_{ik} - \tilde m^{(j)}}
$$

The running statistics after block $j$ satisfy

$$
\begin{aligned}
m_i^{(j)}    &= \max\bigl(m_i^{(j-1)},\, \tilde m^{(j)}\bigr) \\
\alpha       &= \exp\bigl(m_i^{(j-1)} - m_i^{(j)}\bigr), \quad
\beta        = \exp\bigl(\tilde m^{(j)} - m_i^{(j)}\bigr) \\
\ell_i^{(j)} &= \alpha\, \ell_i^{(j-1)} + \beta\, \tilde\ell^{(j)}
\end{aligned}
$$

The accumulator has two common forms:

**Normalized** (used in this repo's v1):

$$
O_i^{(j)} \;=\; \frac{\alpha\, \ell_i^{(j-1)}\, O_i^{(j-1)} \;+\; \beta\, \bigl(e^{S^{(j)} - \tilde m^{(j)}}\bigr) V^{(j)}}{\ell_i^{(j)}}
$$

**Unnormalized** (Dao 2023): keep $\tilde O_i^{(j)} = \alpha\, \tilde O_i^{(j-1)} + \beta\, (e^{S^{(j)} - \tilde m^{(j)}}) V^{(j)}$, divide by $\ell_i^{(T_c)}$ once at the end.

The variants below have identical FLOP counts and final outputs; the differences are memory traffic, register pressure, and which hardware units do the matmuls.

## naive — paper form

Loop order: **$j$ outer, $i$ inner**. Per query row, $m_i$, $\ell_i$, and $O_i$ live in HBM; each $j$-tile loads them, applies the recurrence, and writes them back. Simple, but every tile pays the gmem read/write cost for $(m, \ell, O)$. Matmuls are scalar FFMA32. One thread per query row inside a CTA.

## register — swapped loops + register residency

Loop order: **$i$ outer, $j$ inner**. Each query block is handled by one CTA; $m_i$, $\ell_i$, and the accumulator $O_i$ stay in registers across the $j$ loop, so only the final $O$ is written to HBM. One warp per query row; row reductions use `__shfl_xor_sync`. Matmuls are still scalar FFMA32.

## mma — tensor-core matmul

Same loop structure as `register`, but both $S = Q K^\top$ and $O \mathrel{+}= P V$ are done by the `SM80_16x8x16_F32F16F16F32_TN` MMA atom (FP16 operands, FP32 accumulator). Thread mapping follows the MMA fragment layout: each of the 32 warp threads owns 4 elements of the 16×8 output tile (two rows × two columns). Row-wise softmax reductions run across the 4-thread group that shares each accumulator row (butterfly `shfl_xor` at offsets 1, 2). Uses the **unnormalized** update: $\tilde O^{(j)} = \alpha\, \tilde O^{(j-1)} + \beta\, P^{(j)} V^{(j)}$ across the $j$ loop, with a single division by $\ell_i^{(T_c)}$ in the epilogue. Smem is a single-buffer `__syncthreads`-gated copy — no pipelining.

## multistage — cp.async pipelined

Identical to `mma` except the gmem → smem copies are issued as `cp.async` (`SM80_CP_ASYNC_CACHEGLOBAL`) into an `NUM_STAGES`-deep ring buffer. The mainloop prefetches the next $K, V$ tile while the current tile's MMA + softmax runs. Critical stall reason targeted is `long_scoreboard` (wait for global loads) — overlapping these with compute is the win.
