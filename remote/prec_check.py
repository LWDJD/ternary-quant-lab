"""Does the reference fold at reduced precision?

The rank-86 boundary gap has p10 = 0.097% of d, and about 10% of decisions
disagree.  That implies my folded values differ from theirs by ~0.1%, which is
far above fp32 accumulation error (~1e-6) but right at bf16's relative step
(2**-8 ~ 0.4%, the source dtype).  So: fold several ways, rank-86, and see which
agreement with the reference's trits is highest.

fp32 is the baseline; if bf16 input rounding beats it, the reference folds in the
source dtype and precision IS the route.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402

QK, FOLD, K = 128, 1024, 86
P = "model.language_model.layers.0."
HFNAMES = {"blk_0_ffn_gate_weight": P + "mlp.gate_proj.weight",
           "blk_0_ffn_up_weight": P + "mlp.up_proj.weight"}


def load_rows(hf_name, rows):
    for shard in sorted(glob.glob("/tmp/models/qwen38-27b/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402


def to_bf16(a):
    """Round-trip through bfloat16, i.e. what storing/reading at source dtype does."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).to(torch.float32).numpy()


def to_fp16(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float16).to(torch.float32).numpy()


s = np.load("/mnt/workspace/prismwork/ref_small.npz")


def rank86_agree(x_full, tgt, rows, cols):
    x_full = np.asarray(x_full)[:, :cols]      # fold_weight returns full width
    nb = cols // QK
    rb = x_full.reshape(rows, nb, QK)
    idx = np.argsort(-np.abs(rb), axis=2)[:, :, :K]
    pred = np.zeros_like(rb)
    np.put_along_axis(pred, idx, np.take_along_axis(np.sign(rb), idx, axis=2), axis=2)
    return float((pred.reshape(rows, cols) == tgt).mean()) * 100


variants = ["fp32 (baseline)", "bf16 input", "bf16 input+output", "fp16 input"]
acc = {v: [] for v in variants}
for base, hf_name in HFNAMES.items():
    tgt = s[base + "__t"]
    rows, cols = tgt.shape
    w = load_rows(hf_name, rows)
    if w is None:
        continue
    width = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")

    w32 = w.astype(np.float32)
    x = PP.fold_weight(w32, sign, FOLD).astype(np.float32)
    acc["fp32 (baseline)"].append(rank86_agree(x, tgt, rows, cols))

    xb = PP.fold_weight(to_bf16(w32), sign, FOLD).astype(np.float32)
    acc["bf16 input"].append(rank86_agree(xb, tgt, rows, cols))

    xbo = to_bf16(PP.fold_weight(to_bf16(w32), sign, FOLD).astype(np.float32))
    acc["bf16 input+output"].append(rank86_agree(xbo, tgt, rows, cols))

    xh = PP.fold_weight(to_fp16(w32), sign, FOLD).astype(np.float32)
    acc["fp16 input"].append(rank86_agree(xh, tgt, rows, cols))
    print(f"  {base}: done")

print(f"\n{'variant':24s} {'mean agreement':>15s}")
for v in variants:
    if acc[v]:
        print(f"{v:24s} {np.mean(acc[v]):14.2f}%")
print("\nIf a reduced-precision variant beats fp32, the reference folds in that dtype.")
