"""Fit the deployable rule: d = a*mean|x| with the decision at theta*d.

The reference's stored d measures ~1.40*mean|x| and its non-zero fraction is only
reproduced with a boundary near 0.38*d, not 0.5*d.  A threshold and a
reconstruction magnitude are separate knobs, so both have to be fitted:

    trit = 0 if |x| < theta*d else sign(x),   d = a*mean|x|

This sweeps (a, theta) using only quantities computable from the weights, and
scores against the reference's own trits.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

QK, FOLD = 128, 1024
HF = "/tmp/q38-27b"
P = "model.language_model.layers.0."
HFNAMES = {"blk_0_ffn_gate_weight": P + "mlp.gate_proj.weight",
           "blk_0_ffn_up_weight": P + "mlp.up_proj.weight",
           "blk_0_ssm_out_weight": P + "linear_attn.out_proj.weight"}


def load_rows(hf_name, rows):
    for shard in sorted(glob.glob(f"{HF}/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


s = np.load("/mnt/workspace/prismwork/ref_small.npz")
loaded = []
for base, hf_name in HFNAMES.items():
    tgt, dref = s[base + "__t"], s[base + "__d"].astype(np.float32)
    rows, cols = tgt.shape
    w = load_rows(hf_name, rows)
    if w is None:
        print(f"  {base}: HF missing")
        continue
    full = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{full}.npy")
    fd = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :cols]
    loaded.append((base, fd, tgt, dref))
    print(f"  {base}: folded {fd.shape}, ref nonzero {float((tgt != 0).mean()) * 100:.2f}%")

print(f"\n{'a':>5s} {'theta':>6s} {'nonzero':>8s} {'agree':>8s}")
total = sum(t.size for _, _, t, _ in loaded)
best = (-1.0, None, None)
for a in (1.30, 1.35, 1.38, 1.40, 1.42, 1.45, 1.50):
    for th in (0.34, 0.36, 0.38, 0.40, 0.44, 0.50):
        ag = nz = 0.0
        for _, fd, tgt, _ in loaded:
            rows, cols = fd.shape
            nb = cols // QK
            rb = fd.reshape(rows, nb, QK)
            tg = tgt.reshape(rows, nb, QK)
            # d is per QUANTISATION block (128), not per row
            d = np.maximum(np.float32(a) * np.abs(rb).mean(axis=2), 1e-30)
            pred = np.where(np.abs(rb) >= th * d[:, :, None], np.sign(rb), 0).astype(np.int8)
            ag += float((pred == tg).mean()) * tgt.size
            nz += float((pred != 0).mean()) * tgt.size
        ag, nz = ag / total, nz / total
        flag = ""
        if ag > best[0]:
            best = (ag, a, th)
            flag = "  <- best agreement"
        if abs(nz - 0.672) < 0.008:
            flag += "  [matches 67.2%]"
        print(f"{a:5.2f} {th:6.2f} {nz * 100:7.2f}% {ag * 100:7.2f}%{flag}")

print(f"\nbest agreement {best[0] * 100:.2f}% at a={best[1]} theta={best[2]}")
print("reference: nonzero 67.2%. The shipped encoder (d=amax, theta=0.5) scores near chance.")
