"""Final check: is the rule "keep the k largest |x| per 128-block"?

The per-block non-zero count is pinned at 86 (sd 0.35 vs binomial 5.31), which is
a rank rule, not a threshold.  The fold is already verified exact (100% sign
agreement), so this isolates the decision completely:

    keep the k largest magnitudes in each block, zero the rest, sign(x)*d

Sweeps k so the answer is not assumed.
"""
from __future__ import annotations

import glob
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


def keep_top_k(rb: np.ndarray, k: int) -> np.ndarray:
    """rb: (rows, nb, QK). Zero all but the k largest magnitudes in each block."""
    idx = np.argsort(-np.abs(rb), axis=2)[:, :, :k]
    out = np.zeros_like(rb)
    np.put_along_axis(out, idx, np.take_along_axis(rb, idx, axis=2), axis=2)
    return out


s = np.load("/mnt/workspace/prismwork/ref_small.npz")
loaded = []
for base, hf_name in HFNAMES.items():
    tgt, dref = s[base + "__t"], s[base + "__d"].astype(np.float32)
    w = load_rows(hf_name, tgt.shape[0])
    full = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{full}.npy")
    fd = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :tgt.shape[1]]
    loaded.append((base, fd, tgt))
    print(f"  {base}: {fd.shape}  ref nonzero {float((tgt != 0).mean()) * 100:.2f}%")

total = sum(t.size for _, _, t in loaded)
print(f"\n{'k':>4s} {'nonzero':>8s} {'agreement':>10s}")
best = (-1, None)
for k in (80, 84, 85, 86, 87, 88, 90, 96):
    hit = nz = 0
    for _, fd, tgt in loaded:
        rows, cols = fd.shape
        rb = fd.reshape(rows, cols // QK, QK)
        pred = np.sign(keep_top_k(rb, k)).astype(np.int8)
        hit += int((pred.reshape(rows, cols) == tgt).sum())
        nz += int((pred != 0).sum())
    print(f"{k:4d} {nz / total * 100:7.2f}% {hit / total * 100:9.2f}%")
    if hit > best[0]:
        best = (hit, k)

print(f"\nbest: k={best[1]}  agreement={best[0] / total * 100:.2f}%")
print("reference: 86 per block, 67.2% non-zero. Chance is 33.3%.")
