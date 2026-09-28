"""What function of the rotated 128-group does the reference's scale d equal?

Setup.  Both the reference and our packer quantise the *folded* weights, and the
fold is fully determined: the runtime builds the normalized Sylvester Hadamard
itself and the reference stores its sign vectors in the GGUF.  So for a tensor we
have everything needed to reconstruct the rotation-domain values that the
reference actually saw:

    W_rot = fold(W_true, signs_ref, block)

The 128-value quantisation groups run along the input dimension, and that layout is
the same in the GGUF, so reference group (row, g) pairs with W_rot[row, 128g:128g+128].

Then the question is one-dimensional per group: d_ref = f(|x|) for what f?  Report
the spread of several candidate ratios -- a constant one names the rule.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")

import prism_pack as PP  # noqa: E402
from gguf import GGUFReader  # noqa: E402

REF = "/tmp/bonsai2-PTQ1_0.gguf"
QK = 128
BLOCK = 1024

# GGUF name -> HF safetensors name (Qwen3.8-27B / Bonsai 2 layout)
TENSORS = [
    ("blk.0.ffn_gate.weight", "model.language_model.layers.0.mlp.gate_proj.weight"),
    ("blk.0.ffn_up.weight",   "model.language_model.layers.0.mlp.up_proj.weight"),
    ("blk.0.ffn_down.weight", "model.language_model.layers.0.mlp.down_proj.weight"),
    ("blk.1.ffn_gate.weight", "model.language_model.layers.1.mlp.gate_proj.weight"),
    ("blk.3.attn_q.weight",   "model.language_model.layers.3.self_attn.q_proj.weight"),
]


def ref_signs(r):
    widths = [int(v) for v in r.fields["prism.hadamard.sign_widths"].contents()]
    vals = np.array([int(v) for v in r.fields["prism.hadamard.sign_values"].contents()],
                    dtype=np.float32)
    out, off = {}, 0
    for w in widths:
        out[w] = vals[off:off + w]
        off += w
    return out


def ref_scales(r, name):
    """Per-128-group d from the reference, as (nrow, ngroups)."""
    t = next(x for x in r.tensors if x.name == name)
    ne = list(t.shape)              # ggml order: [in, out]
    nin, nout = int(ne[0]), int(ne[1])
    raw = t.data
    ngrp = nin // QK
    b = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=np.uint8)
    b = b.reshape(nout, ngrp, 28)
    d = b[:, :, 26:28].copy().view(np.float16).reshape(nout, ngrp).astype(np.float32)
    return d, nin, nout


def load_hf(weight_name):
    """Read one tensor.  safetensors numpy mode cannot represent bfloat16, so go
    through torch and cast down."""
    import glob
    import torch
    from safetensors import safe_open
    for shard in glob.glob("/tmp/q38-27b/*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            if weight_name in list(f.keys()):
                return f.get_tensor(weight_name).to(torch.float32).numpy()
    return None


r = GGUFReader(REF)
signs = ref_signs(r)
print(f"reference sign widths: {sorted(signs)}")

for gname, hname in TENSORS:
    if not any(t.name == gname for t in r.tensors):
        print(f"\n--- {gname}: not in reference")
        continue
    d_ref, nin, nout = ref_scales(r, gname)
    w = load_hf(hname)
    if w is None:
        print(f"\n--- {gname}: HF tensor {hname} not found yet (download running?)")
        continue
    # HF is [out, in]; fold expects the input dim last
    x = np.asarray(w, dtype=np.float32)
    if x.shape[-1] != nin:
        x = x.T
    assert x.shape == (nout, nin), f"{gname}: {x.shape} vs ref ({nout},{nin})"
    sgn = signs.get(nin)
    if sgn is None:
        print(f"\n--- {gname}: no reference sign vector for width {nin}")
        continue
    xr = PP.fold_weight(x, sgn, BLOCK).astype(np.float32).reshape(nout, nin // QK, QK)
    dr = d_ref

    a = np.abs(xr)
    stats = {
        "max":  a.max(axis=2),
        "mean": a.mean(axis=2),
        "rms":  np.sqrt((xr ** 2).mean(axis=2)),
        "std":  xr.std(axis=2),
        "q90":  np.quantile(a, 0.90, axis=2),
        "q75":  np.quantile(a, 0.75, axis=2),
    }
    print(f"\n--- {gname}  shape {x.shape}  groups {dr.size}")
    print(f"    d_ref: mean={dr.mean():.6g} min={dr.min():.6g} max={dr.max():.6g}")
    best = None
    for k, v in stats.items():
        ratio = np.where(v > 0, dr / np.maximum(v, 1e-30), np.nan)
        ok = np.isfinite(ratio)
        cv = float(np.nanstd(ratio[ok]) / max(abs(np.nanmean(ratio[ok])), 1e-30))
        print(f"    d/{k:4s}: ratio mean={np.nanmean(ratio):.5f}  cv={cv:.4f}")
        if best is None or cv < best[1]:
            best = (k, cv, float(np.nanmean(ratio)))
    print(f"    -> most constant: d/{best[0]} = {best[2]:.5f}  (cv={best[1]:.4f})")
    nz = float((np.abs(xr / dr[:, :, None]) > 0.5).mean())
    print(f"    non-zero fraction at d_ref: {nz * 100:.2f}%")
