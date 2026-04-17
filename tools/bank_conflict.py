#!/usr/bin/env python3

"""
Bank conflict analyzer for CuTe MMA layouts.

Reproduces CuTe's layout algebra in pure Python to trace every step from
MMA thread/value layouts through shared-memory addressing to bank
assignments.

  analyze(T, V, atom_layout, smem_layout, bits, swizzle=None)

Two separate layouts are required:

  atom_layout = shape
      MMA atom tile shape.  Element index -> (row, col) via column-major
      idx2crd.  Only shape needed, no stride.  Fixed by hardware ISA.
      e.g. A: (16, 16)  -- (M, K)
           B: (8, 16)   -- (N, K)

  smem_layout = (shape, stride)
      Shared memory tile layout.  Maps (row, col) logical coordinates ->
      physical smem element offset.  Depends on kernel tile sizes (BM, BN, BK).
      e.g. sA: ((128, 32), (32, 1))  -- BM=128, BK=32, K-contiguous
           sB: ((128, 32), (32, 1))  -- BN=128, BK=32, K-contiguous

Layout convention follows CuTe:
  Shape and Stride may be nested tuples.
  Indexing is column-major (least-significant sub-mode first).

Example (SM80_16x8x16_F32F16F16F32_TN, A-side):

    T = ((4, 8), (32, 1))
    V = ((2, 2, 2), (16, 8, 128))
    atom = (16, 16)                  # (M, K) atom shape
    smem = ((128, 32), (32, 1))      # smem tile: BM=128, BK=32
    analyze(T, V, atom, smem, bits=16, swizzle=(2, 3, 6))
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# CuTe layout helpers (pure Python, column-major)
# ---------------------------------------------------------------------------

def _flatten(x):
    """Recursively flatten nested tuples/lists into a flat list."""
    if isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [x]


def _size(shape):
    """Total number of elements in a (possibly nested) shape."""
    s = 1
    for d in _flatten(shape):
        s *= d
    return s


def _crd2idx(coord, shape, stride):
    """CuTe crd2idx: hierarchical (coord, shape, stride) -> flat index."""
    if isinstance(shape, (tuple, list)):
        idx = 0
        for c, s, st in zip(coord, shape, stride):
            idx += _crd2idx(c, s, st)
        return idx
    return coord * stride


def _idx2crd(idx, shape):
    """CuTe idx2crd: linear index -> column-major coordinate tuple."""
    flat = _flatten(shape)
    coords_flat = []
    rem = idx
    for d in flat:
        coords_flat.append(rem % d)
        rem //= d
    return _unflatten(coords_flat, shape)


def _unflatten(flat_list, shape):
    """Rebuild the nesting structure of shape from a flat list."""
    if isinstance(shape, (tuple, list)):
        result = []
        offset = 0
        for s in shape:
            n = len(_flatten(s))
            result.append(_unflatten(flat_list[offset:offset + n], s))
            offset += n
        return tuple(result)
    return flat_list[0]


def layout_eval(layout, linear_idx):
    """Evaluate a CuTe layout at a linear index."""
    shape, stride = layout
    coord = _idx2crd(linear_idx, shape)
    return _crd2idx(coord, shape, stride)


# ---------------------------------------------------------------------------
# Swizzle
# ---------------------------------------------------------------------------

def swizzle_fn(flat_idx, B, M, S):
    """Apply Swizzle<B,M,S>: XOR bits [S+B-1:S] into bits [M+B-1:M]."""
    src_bits = (flat_idx >> S) & ((1 << B) - 1)
    return flat_idx ^ (src_bits << M)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RED   = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _flat_to_bank(flat_elem, bytes_per_elem):
    return (flat_elem * bytes_per_elem // 4) % 32


def _flat_to_word(flat_elem, bytes_per_elem):
    return flat_elem * bytes_per_elem // 4


def _fmt(coord):
    if isinstance(coord, (tuple, list)):
        return "(" + ",".join(_fmt(c) for c in coord) + ")"
    return str(coord)


def _print_table(headers, rows):
    col_widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))
    fmt = "  ".join(f"{{:>{w}}}" for w in col_widths)
    print(fmt.format(*[str(h) for h in headers]))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for r in rows:
        print(fmt.format(*[str(v) for v in r]))


def _build_coord_map(layout):
    """Invert a CuTe layout: build offset → coordinate mapping.

    For layout (shape, stride), returns dict: layout(coord) → flat_coord_tuple.
    Thread/value id IS the offset (threadIdx.x or value index), so this gives
    tid → (tm, tk, ...) or vid → (vm, vk, ...).
    """
    shape, stride = layout
    total = _size(shape)
    inv = {}
    for i in range(total):
        coord = _idx2crd(i, shape)
        offset = _crd2idx(coord, shape, stride)
        inv[offset] = tuple(_flatten(coord)) if isinstance(coord, tuple) else (coord,)
    return inv


def _tiled_smem_offset(coord, atom_shape, atom_stride, tile_shape, swizzle=None):
    """Evaluate tile_to_shape(composition(Swizzle, AtomLayout), TileShape) at coord.

    Implements: offset = Swizzle(AtomStride · coord_local) + tile_linear × atom_cosize
    """
    # Split into local coord within atom and tile index
    local = tuple(c % a for c, a in zip(coord, atom_shape))
    tile_idx = tuple(c // a for c, a in zip(coord, atom_shape))

    # Local offset via atom stride
    local_offset = sum(l * s for l, s in zip(local, atom_stride))

    # Apply swizzle
    if swizzle:
        local_offset = swizzle_fn(local_offset, *swizzle)

    # Atom cosize
    atom_cosize = sum((a - 1) * s for a, s in zip(atom_shape, atom_stride)) + 1

    # Tile linear index (colexicographic)
    tiles = tuple(t // a for t, a in zip(tile_shape, atom_shape))
    tile_linear = 0
    multiplier = 1
    for ti, nt in zip(tile_idx, tiles):
        tile_linear += ti * multiplier
        multiplier *= nt

    return local_offset + tile_linear * atom_cosize


def _worst_conflict(word_map, num_threads, num_values):
    """Return worst-case N-way conflict across all value ids.

    Conflict = max distinct 32-bit words hitting the same bank for any vid.
    Same-word accesses are broadcasts (free).
    """
    worst = 1
    for vid in range(num_values):
        bank_to_words: dict[int, set[int]] = {}
        for tid in range(num_threads):
            b = (word_map[(tid, vid)] % 32)
            bank_to_words.setdefault(b, set()).add(word_map[(tid, vid)])
        worst = max(worst, max(len(ws) for ws in bank_to_words.values()))
    return worst


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(T, V, atom_layout, smem_layout, bits=16, swizzle=None):
    """Unified bank conflict analysis.

    Parameters
    ----------
    T : (shape, stride)  -- thread mode of MMA layout
    V : (shape, stride)  -- value mode of MMA layout
    atom_layout : tuple (shape only)
        MMA atom tile shape.  Element index -> (row, col) via column-major
        idx2crd.  Only shape matters, stride is never used.
        e.g. A: (16, 16),  B: (8, 16)
    smem_layout : (shape, stride)
        Shared memory tile layout.  (row, col) -> physical smem element
        offset via smem stride.
        e.g. ((128, 32), (32, 1)) for BM=128 or BN=128, BK=32
    bits : int  -- element size in bits (16 for half, 32 for float)
    swizzle : (B, M, S) or None  -- optional Swizzle parameters

    Returns
    -------
    int : worst-case N-way bank conflict (1 = no conflict)
    """
    bpe = bits // 8
    atom_shape = atom_layout
    smem_shape, smem_stride = smem_layout
    num_threads = _size(T[0])
    num_values = _size(V[0])

    sw_str = "no swizzle"
    if swizzle:
        B_sw, M_sw, S_sw = swizzle
        sw_str = f"Swizzle<{B_sw},{M_sw},{S_sw}>"

    print(f"\n{_BOLD}Bank Conflict Analysis ({sw_str}){_RESET}")
    print(f"  T: shape={T[0]}, stride={T[1]}")
    print(f"  V: shape={V[0]}, stride={V[1]}")
    print(f"  atom: shape={atom_shape}")
    print(f"  smem: shape={smem_shape}, stride={smem_stride}")
    if swizzle:
        print(f"  {sw_str}: XOR bits [{S_sw+B_sw-1}:{S_sw}] "
              f"into [{M_sw+B_sw-1}:{M_sw}]")
    print(f"  {bits}-bit elements, {num_threads} threads, "
          f"{num_values} values/thread")

    # Build rows — always compute no-swizzle; add swizzle columns when provided
    if swizzle:
        headers = ("tid", "vid", "element", "atom_coord",
                   "smem_flat", "word", "bank",
                   "sw_flat", "sw_word", "sw_bank")
    else:
        headers = ("tid", "vid", "element", "atom_coord",
                   "smem_flat", "word", "bank")

    rows = []
    word_map_nosw = {}
    word_map_sw = {}

    for vid in range(num_values):
        v_elem = layout_eval(V, vid)
        for tid in range(num_threads):
            t_elem = layout_eval(T, tid)
            elem = t_elem + v_elem

            # Step 1: element -> (row, col) via atom layout
            atom_coord = _idx2crd(elem, atom_shape)

            # Step 2: (row, col) -> physical smem offset via smem stride
            smem_flat = sum(
                c * s for c, s in zip(_flatten(atom_coord),
                                      _flatten(smem_stride)))
            word = _flat_to_word(smem_flat, bpe)
            bank = _flat_to_bank(smem_flat, bpe)
            word_map_nosw[(tid, vid)] = word

            if swizzle:
                sw_flat = swizzle_fn(smem_flat, *swizzle)
                sw_word = _flat_to_word(sw_flat, bpe)
                sw_bank = _flat_to_bank(sw_flat, bpe)
                word_map_sw[(tid, vid)] = sw_word
                rows.append((tid, vid, elem, _fmt(atom_coord),
                             smem_flat, word, bank,
                             sw_flat, sw_word, sw_bank))
            else:
                rows.append((tid, vid, elem, _fmt(atom_coord),
                             smem_flat, word, bank))

    _print_table(headers, rows)

    worst_nosw = _worst_conflict(word_map_nosw, num_threads, num_values)
    if swizzle:
        worst_sw = _worst_conflict(word_map_sw, num_threads, num_values)
        def _cf(w):
            if w == 1:
                return f"{_GREEN}{w}-way (no bank conflicts){_RESET}"
            return f"{_RED}{w}-way bank conflict{_RESET}"
        print(f"\n  no swizzle: {_cf(worst_nosw)}")
        print(f"  {sw_str}:  {_cf(worst_sw)}")
        return worst_nosw, worst_sw
    else:
        if worst_nosw == 1:
            print(f"\n{_GREEN}{worst_nosw}-way (no bank conflicts){_RESET}")
        else:
            print(f"\n{_RED}{worst_nosw}-way bank conflict{_RESET}")
        return worst_nosw


# ---------------------------------------------------------------------------
# Pre-built MMA layouts (from mma_traits_sm80.hpp)
# ---------------------------------------------------------------------------

# always assume (M,K) for A and (N,K) for B in atom
SM80_16x8x16_A = {
    "T": ((4, 8), (32, 1)),            # thread mode
    "V": ((2, 2, 2), (16, 8, 128)),    # value mode
    "atom": (16, 16),                   # (M, K) atom shape
}
SM80_16x8x16_B = {
    "T": ((4, 8), (16, 1)),
    "V": ((2, 2), (8, 64)),
    "atom": (8, 16),                    # (N, K) atom shape
}


# ---------------------------------------------------------------------------
# G2S TiledCopy analysis
# ---------------------------------------------------------------------------

def g2s_analyze(thr_layout, val_layout, smem_atom_shape, smem_atom_stride,
                tile_shape, bits=16, swizzle=None):
    """Bank conflict analysis for gmem→smem TiledCopy.

    CuTe formulation — make_tiled_copy(CopyAtom, T, V):

        T_coord = T⁻¹(tid)           — invert thread layout
        V_coord = V⁻¹(vid)           — invert value layout
        tile_coord[i] = T_coord[i] × V.shape[i] + V_coord[i]   — blocked product
        offset(tid,vid) = tile_to_shape(Swizzle ∘ AtomLayout, TileShape)(tile_coord)

    For wide (128-bit) stores, hardware issues 4 phases of 32-bit words.
    Bank conflicts are per-phase within each warp.

    Parameters
    ----------
    thr_layout       : (shape, stride)  — thread layout from make_tiled_copy
    val_layout       : (shape, stride)  — value layout from make_tiled_copy
    smem_atom_shape  : tuple  — swizzle atom shape, e.g. (16, 32)
    smem_atom_stride : tuple  — atom stride (LayoutRight), e.g. (32, 1)
    tile_shape       : tuple  — full smem tile, e.g. (128, 32)
    bits             : int    — element bits (16 for half_t)
    swizzle          : (B, M, S) or None

    Returns
    -------
    int (no swizzle) or (int, int) (with swizzle) : worst-case N-way conflict
    """
    bpe = bits // 8
    num_thr = _size(thr_layout[0])
    num_val = _size(val_layout[0])
    val_flat_shape = tuple(_flatten(val_layout[0]))
    warp_size = 32

    sw_str = f"Swizzle<{swizzle[0]},{swizzle[1]},{swizzle[2]}>" if swizzle else "no swizzle"

    print(f"\n{_BOLD}G2S TiledCopy Bank Conflict Analysis ({sw_str}){_RESET}")
    print(f"  ThreadLayout: shape={thr_layout[0]}, stride={thr_layout[1]}")
    print(f"  ValueLayout:  shape={val_layout[0]}, stride={val_layout[1]}")
    print(f"  smem atom:    shape={smem_atom_shape}, stride={smem_atom_stride}")
    print(f"  tile shape:   {tile_shape}, {sw_str}")
    if swizzle:
        B_sw, M_sw, S_sw = swizzle
        print(f"  {sw_str}: XOR bits [{S_sw+B_sw-1}:{S_sw}] "
              f"into [{M_sw+B_sw-1}:{M_sw}]")
    print(f"  {bits}-bit elements, {num_thr} threads, "
          f"{num_val} values/thread (warp 0 shown)")

    # Build inverse maps: tid → (tm, tk, ...), vid → (vm, vk, ...)
    thr_coord_of = _build_coord_map(thr_layout)
    val_coord_of = _build_coord_map(val_layout)

    # --- Build table rows (warp 0 only, like analyze()) ---
    if swizzle:
        headers = ("tid", "vid", "thr_coord", "tile_coord",
                   "smem_flat", "word", "bank",
                   "sw_flat", "sw_word", "sw_bank")
    else:
        headers = ("tid", "vid", "thr_coord", "tile_coord",
                   "smem_flat", "word", "bank")

    rows = []
    word_map_nosw = {}
    word_map_sw = {}

    for vid in range(num_val):
        vc = val_coord_of[vid]
        for tid in range(warp_size):          # warp 0 only
            tc = thr_coord_of[tid]
            tile_coord = tuple(t * vs + v
                               for t, v, vs in zip(tc, vc, val_flat_shape))

            # No-swizzle offset
            smem_flat = _tiled_smem_offset(
                tile_coord, smem_atom_shape, smem_atom_stride, tile_shape, None)
            word = _flat_to_word(smem_flat, bpe)
            bank = _flat_to_bank(smem_flat, bpe)
            word_map_nosw[(tid, vid)] = word

            if swizzle:
                sw_flat = _tiled_smem_offset(
                    tile_coord, smem_atom_shape, smem_atom_stride,
                    tile_shape, swizzle)
                sw_word = _flat_to_word(sw_flat, bpe)
                sw_bank = _flat_to_bank(sw_flat, bpe)
                word_map_sw[(tid, vid)] = sw_word
                rows.append((tid, vid, _fmt(tc), _fmt(tile_coord),
                             smem_flat, word, bank,
                             sw_flat, sw_word, sw_bank))
            else:
                rows.append((tid, vid, _fmt(tc), _fmt(tile_coord),
                             smem_flat, word, bank))

    _print_table(headers, rows)

    # --- Conflict summary (same style as analyze()) ---
    worst_nosw = _worst_conflict(word_map_nosw, warp_size, num_val)
    if swizzle:
        worst_sw = _worst_conflict(word_map_sw, warp_size, num_val)
        def _cf(w):
            if w == 1:
                return f"{_GREEN}{w}-way (no bank conflicts){_RESET}"
            return f"{_RED}{w}-way bank conflict{_RESET}"
        print(f"\n  no swizzle: {_cf(worst_nosw)}")
        print(f"  {sw_str}:  {_cf(worst_sw)}")
        return worst_nosw, worst_sw
    else:
        if worst_nosw == 1:
            print(f"\n{_GREEN}{worst_nosw}-way (no bank conflicts){_RESET}")
        else:
            print(f"\n{_RED}{worst_nosw}-way bank conflict{_RESET}")
        return worst_nosw


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def demo_g2s_A(BM=128, BK=32):
    """G2S copy bank conflict analysis for sA."""
    NUM_THREADS = 128
    VEC = 8
    BK_VEC = BK // VEC           # 4
    ThrM = NUM_THREADS // BK_VEC  # 32

    print("=" * 72)
    print(f"G2S Copy A  |  sA({BM},{BK}), {NUM_THREADS} threads, Vec128  |  FP16")
    print("=" * 72)
    thr_layout = ((ThrM, BK_VEC), (BK_VEC, 1))   # LayoutRight
    val_layout = ((1, VEC), (0, 1))

    # smem: tile_to_shape(Swizzle<2,3,6> ∘ Layout((16,BK), LayoutRight), (BM,BK))
    g2s_analyze(thr_layout, val_layout,
                smem_atom_shape=(16, BK), smem_atom_stride=(BK, 1),
                tile_shape=(BM, BK), bits=16, swizzle=(2, 3, 6))


def demo_g2s_B(BN=128, BK=32):
    """G2S copy bank conflict analysis for sB."""
    NUM_THREADS = 128
    VEC = 8
    BK_VEC = BK // VEC
    ThrN = NUM_THREADS // BK_VEC

    print("\n" + "=" * 72)
    print(f"G2S Copy B  |  sB({BN},{BK}), {NUM_THREADS} threads, Vec128  |  FP16")
    print("=" * 72)

    thr_layout = ((ThrN, BK_VEC), (BK_VEC, 1))
    val_layout = ((1, VEC), (0, 1))

    # sB atom is (8, BK) — from the kernel's Swizzle atom
    g2s_analyze(thr_layout, val_layout,
                smem_atom_shape=(8, BK), smem_atom_stride=(BK, 1),
                tile_shape=(BN, BK), bits=16, swizzle=(2, 3, 6))


def demo_sm80_A(BM=128, BK=32):
    print("=" * 72)
    print(f"SM80_16x8x16 A-operand  |  smem ({BM},{BK}):({BK},1)  |  FP16")
    print("=" * 72)
    T = SM80_16x8x16_A["T"]
    V = SM80_16x8x16_A["V"]
    atom = SM80_16x8x16_A["atom"]
    smem = ((BM, BK), (BK, 1))
    analyze(T, V, atom, smem, bits=16, swizzle=(2, 3, 6))


def demo_sm80_B(BN=128, BK=32):
    print("\n" + "=" * 72)
    print(f"SM80_16x8x16 B-operand  |  smem ({BN},{BK}):({BK},1)  |  FP16")
    print("=" * 72)
    T = SM80_16x8x16_B["T"]
    V = SM80_16x8x16_B["V"]
    atom = SM80_16x8x16_B["atom"]
    smem = ((BN, BK), (BK, 1))
    analyze(T, V, atom, smem, bits=16, swizzle=(2, 3, 6))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Bank conflict analyzer for CuTe MMA and G2S copy layouts")
    parser.add_argument("operand", nargs="?", default="AB",
                        help="Which operand to demo: A, B, or AB (default: AB)")
    parser.add_argument("--g2s", action="store_true",
                        help="Analyze gmem→smem copy (default: smem→register MMA)")
    args = parser.parse_args()
    if args.g2s:
        if "A" in args.operand.upper():
            demo_g2s_A()
        if "B" in args.operand.upper():
            demo_g2s_B()
    else:
        if "A" in args.operand.upper():
            demo_sm80_A()
        if "B" in args.operand.upper():
            demo_sm80_B()
