#!/usr/bin/env python3
"""
PrismML Hadamard packing: weight folding + manifest emission.

Contract (from prism-tip, PrismML llama.cpp fork, build prism-b10743 adfffbe):

  whitepaper bonsai-2-27b §2.4:
      R = (1/sqrt(n)) H_n S,   n = block size (1024 for Bonsai 2 27B)
      f(x) = W(R x)
  "Blockwise Hadamard rotation (block 1024, fixed +/-1 signs)"

  runtime, src/llama-graph.cpp build_lora_mm / build_lora_mm_id:
      cur_mm = ggml_mul(cur_mm, t.signs);                  // x <- S x
      cur_mm = llama_mul_mat_hadamard(ctx0, cur_mm, t.rot); // x <- H_norm x
      res    = ggml_mul_mat(w, cur_mm);                     // y = W_stored (H_norm S x)

  src/llama-model.cpp builds the rotation exactly as:
      scale = 1/sqrt(block_size)
      data[row*bs + col] = (parity(row & col) & 1) ? -scale : scale
  i.e. H_norm = H_raw / sqrt(n) with H_raw[i][j] = (-1)^popcount(i & j) (Sylvester order).
  H_norm and S are both involutions, so the fold is exactly:

      W_stored = W @ S @ H_norm        (blockwise along the input dimension)

  and W_stored @ (H_norm @ (s * x)) == W @ x  for every x.  No approximation.

Manifest schema (conversion/base.py add_hadamard_metadata):
    { "schema_version": 1|2|3, "kind": "hadamard-weight-fold",
      "status": "requires-matching-runtime",
      "transform": {"name": "normalized-signed-sylvester-walsh-hadamard",
                    "block_size": n, "sign_mode": "identity"|"explicit"},
      "signs": {"<width>": [.. +/-1 ..]},         # explicit only
      "tensors": [{"name": <hf name>, "axis": -1, "role": "fold-before-matmul"}],
      "tied_output": bool }                        # schema 3 only
"""

import json
import math
import sys

import numpy as np


# ----------------------------------------------------------------------------
# reference construction: byte-for-byte what the runtime does
# ----------------------------------------------------------------------------

def runtime_rot(n: int) -> np.ndarray:
    """Exact copy of the rotation matrix built in src/llama-model.cpp:2047-2058."""
    scale = 1.0 / math.sqrt(n)
    out = np.empty((n, n), dtype=np.float64)
    for row in range(n):
        for col in range(n):
            parity = row & col
            parity ^= parity >> 16
            parity ^= parity >> 8
            parity ^= parity >> 4
            parity ^= parity >> 2
            parity ^= parity >> 1
            out[row, col] = -scale if (parity & 1) else scale
    return out


def fwht_norm(a: np.ndarray) -> np.ndarray:
    """
    Fast normalized Walsh-Hadamard transform along the last axis, Sylvester order.

    Matches H_raw[i][j] = (-1)^popcount(i & j) divided by sqrt(n), which is the
    ordering the runtime's parity loop generates.
    """
    a = np.array(a, dtype=np.float64, copy=True)
    n = a.shape[-1]
    if n & (n - 1):
        raise ValueError(f"length {n} is not a power of two")
    h = 1
    while h < n:
        # view as (..., n/(2h), 2, h); one copy plus two in-place ops keeps the
        # temporaries down -- this runs 9 times over GB-scale arrays
        v = a.reshape(a.shape[:-1] + (n // (2 * h), 2, h))
        u = v[..., 0, :].copy()
        np.add(u, v[..., 1, :], out=v[..., 0, :])
        np.subtract(u, v[..., 1, :], out=v[..., 1, :])
        h *= 2
    return a / math.sqrt(n)


# ----------------------------------------------------------------------------
# folding
# ----------------------------------------------------------------------------

def fold_weight(w: np.ndarray, signs: np.ndarray, block: int) -> np.ndarray:
    """
    W_stored = W @ S @ H_norm, blockwise along the input dimension (last axis).

    w      : [out, in] (in must be a multiple of `block`)
    signs  : [in] entries in {-1, +1}
    """
    in_dim = w.shape[-1]
    if in_dim % block:
        raise ValueError(f"input dim {in_dim} is not a multiple of block {block}")
    if signs.shape[-1] != in_dim:
        raise ValueError(f"sign vector length {signs.shape[-1]} != input dim {in_dim}")

    # step 1: column scale by the signs  (x <- S x happens first at runtime)
    out = w * signs.reshape((1,) * (w.ndim - 1) + (-1,))

    # step 2: blockwise normalized Hadamard along the input dimension
    out = out.reshape(out.shape[:-1] + (in_dim // block, block))
    out = fwht_norm(out)
    return out.reshape(w.shape)


def apply_runtime(x: np.ndarray, signs: np.ndarray, block: int) -> np.ndarray:
    """What the runtime does to the activation: blockwise H_norm applied to (s * x)."""
    x = x * signs
    x = x.reshape(x.shape[:-1] + (x.shape[-1] // block, block))
    return fwht_norm(x).reshape(signs.shape)


# ----------------------------------------------------------------------------
# self-check: prove the fold is exact against the runtime's own R
# ----------------------------------------------------------------------------

def selfcheck(block: int = 8, widths=(16, 64), trials: int = 4) -> int:
    rng = np.random.default_rng(0xC0FFEE)
    fails = 0

    print(f"[1] fast FWHT == runtime parity matrix   (n={block})")
    H = runtime_rot(block)
    eye = np.eye(block)
    fast = fwht_norm(eye)
    d = np.abs(fast - H).max()
    print(f"    max|fwht(I) - runtime_rot| = {d:.3e}   {'OK' if d < 1e-12 else 'MISMATCH'}")
    fails += 0 if d < 1e-12 else 1

    print(f"[2] H_norm is an involution and symmetric")
    d1 = np.abs(H @ H - np.eye(block)).max()
    d2 = np.abs(H - H.T).max()
    print(f"    max|H H - I| = {d1:.3e}   max|H - H^T| = {d2:.3e}   "
          f"{'OK' if max(d1, d2) < 1e-12 else 'MISMATCH'}")
    fails += 0 if max(d1, d2) < 1e-12 else 1

    for in_dim in widths:
        out_dim = 3
        for t in range(trials):
            W = rng.standard_normal((out_dim, in_dim))
            s = rng.choice([-1.0, 1.0], size=in_dim)
            x = rng.standard_normal(in_dim)

            Wp = fold_weight(W, s, block)
            y_ref = W @ x
            y_run = Wp @ apply_runtime(x, s, block)
            err = np.abs(y_ref - y_run).max()
            rel = err / max(1e-12, np.abs(y_ref).max())
            ok = rel < 1e-12
            if not ok:
                fails += 1
            print(f"[3] W'@(H@(s*x)) == W@x   in={in_dim:4d} trial={t}  "
                  f"rel err = {rel:.3e}  {'OK' if ok else 'FAIL'}")

    print("[4] control: wrong fold order (H then S) must NOT match")
    in_dim = widths[0]
    W = rng.standard_normal((3, in_dim))
    s = rng.choice([-1.0, 1.0], size=in_dim)
    x = rng.standard_normal(in_dim)
    # fold as (H then S) instead of the correct (S then H), then run the real runtime transform
    Wbad = W.reshape(3, in_dim // block, block)
    Wbad = fwht_norm(Wbad).reshape(3, in_dim) * s
    rel = np.abs(W @ x - Wbad @ apply_runtime(x, s, block)).max() / np.abs(W @ x).max()
    good = rel > 1e-6
    print(f"    rel err = {rel:.3e}   {'OK (differs, as expected)' if good else 'UNEXPECTED: order does not matter'}")
    fails += 0 if good else 1

    print()
    print("FAILURES:", fails)
    return fails


if __name__ == "__main__":
    if "--selfcheck" in sys.argv or len(sys.argv) == 1:
        sys.exit(1 if selfcheck() else 0)
