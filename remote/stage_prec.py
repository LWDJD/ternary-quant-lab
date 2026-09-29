"""Does the reference round between butterfly stages?

The logical result from §24.3: given identical folded values, any rule monotone in
|x| agrees with theirs except at the single cutoff element.  A 10% disagreement
therefore proves my folded values differ, by roughly the p10 boundary gap (0.097%
of d).  Input rounding (bf16/fp16) did not reproduce it, so if the loss is real it
must happen *between* butterfly stages rather than at the input.

This runs the same butterfly with an optional rounding hook after each stage and
scores rank-86 agreement against the reference's trits.
"""
from __future__ import annotations

import glob
import math
import sys

import numpy as np

QK, FOLD, K = 128, 1024, 86
P = "model.language_model.layers.0."
HFNAMES = {"blk_0_ffn_gate_weight": P + "mlp.gate_proj.weight",
           "blk_0_ffn_up_weight": P + "mlp.up_proj.weight"}
SAMPLE = "/mnt/workspace/prismwork/ref_small.npz"


def round_to(a, mode):
    if mode == "fp32":
        return a
    if mode == "bf16":
        u = a.view(np.uint32)
        r = ((u + np.uint32(0x8000)) & np.uint32(0xFFFF0000)).view(np.float32)
        return np.where(np.isfinite(r), r, a)
    if mode == "fp16":
        return a.astype(np.float16).astype(np.float32)
    raise ValueError(mode)


def butterfly(a, stage_mode):
    """Normalised FWHT along the last axis, optionally rounded after each stage."""
    n = a.shape[-1]
    a = a.astype(np.float32, copy=True)
    h = 1
    while h < n:
        v = a.reshape(a.shape[:-1] + (n // (2 * h), 2, h))
        u = v[..., 0, :].copy()
        np.add(u, v[..., 1, :], out=v[..., 0, :])
        np.subtract(u, v[..., 1, :], out=v[..., 1, :])
        a = round_to(a, stage_mode)
        h *= 2
    return a / math.sqrt(n)


def fold_variant(w, signs, block, stage_mode):
    out = w * signs.reshape((1,) * (w.ndim - 1) + (-1,))
    flat = out.reshape(-1, w.shape[-1])
    blocks = flat.reshape(flat.shape[0], w.shape[-1] // block, block)
    return butterfly(blocks, stage_mode).reshape(flat.shape[0], w.shape[-1])


def rank86(x_full, tgt, rows, cols):
    nb = cols // QK
    rb = np.asarray(x_full)[:, :cols].reshape(rows, nb, QK)
    idx = np.argsort(-np.abs(rb), axis=2)[:, :, :K]
    pred = np.zeros_like(rb)
    np.put_along_axis(pred, idx, np.take_along_axis(np.sign(rb), idx, axis=2), axis=2)
    return float((pred.reshape(rows, cols) == tgt).mean()) * 100


def load_rows(hf_name, rows):
    import torch
    from safetensors import safe_open
    for shard in sorted(glob.glob("/tmp/models/qwen38-27b/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


s = np.load(SAMPLE)
variants = ["fp32", "bf16", "fp16"]
acc = {v: [] for v in variants}
for base, hf_name in HFNAMES.items():
    tgt = s[base + "__t"]
    rows, cols = tgt.shape
    w = load_rows(hf_name, rows)
    if w is None:
        continue
    sign = np.load(f"/mnt/workspace/prismwork/sign_{w.shape[1]}.npy")
    for v in variants:
        x = fold_variant(w, sign, FOLD, v)
        acc[v].append(rank86(x, tgt, rows, cols))
    print(f"  {base}: done")

print(f"\n{'stage rounding':18s} {'agreement':>11s}")
for v in variants:
    if acc[v]:
        print(f"{v:18s} {np.mean(acc[v]):10.2f}%")
print("\nNo variant beating fp32 means the difference is not arithmetic precision,")
print("which rules out the whole reduce-precision family and points at the transform")
print("itself (a different but nearby operator) or at the source values.")
