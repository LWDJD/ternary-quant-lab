"""Closed loop: pack a tensor with the ref rule, decode it, compare to the original.

The no-quant control gives PPL 3.46 against the reference's 4.18, so the reference
pays only 1.21x for 1.75-bit ternary.  A per-weight independent ternary quantiser
should pay far more: for Gaussian weights, keeping 67% gives MSE ~0.21*sigma^2,
i.e. ~46% RMSE, and that compounds through 64 layers.

So measure what my packer actually does.  If the reconstruction error is in the
40-50% range, my quantiser is behaving like the naive ternary it looks like, and no
amount of tuning which weights survive will rescue it -- the reference must be doing
something structurally different.

Compares against the F16 values, i.e. what the no-quant control would have stored.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import ptq1_0 as Q  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

QK, FOLD = 128, 1024
HF = "/tmp/models/qwen38-27b"
P = "model.language_model.layers.0."
JOBS = [("blk.0.ffn_gate.weight", P + "mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", P + "mlp.up_proj.weight"),
        ("blk.0.ffn_down.weight", P + "mlp.down_proj.weight")]

MULT = 1.36645
THETA = 0.38


def load(hf_name, rows):
    for shard in sorted(glob.glob(f"{HF}/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


print(f"{'tensor':28s} {'RMSE/|x|':>10s} {'nonzero':>8s} {'d/mean|x|':>10s}")
allerr = []
for gguf_name, hf_name in JOBS:
    w = load(hf_name, 16)
    if w is None:
        print(f"  {hf_name}: missing")
        continue
    width = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")
    x = PP.fold_weight(w, sign, FOLD).astype(np.float32)

    blob = Q.quantize_blocks_ptq1_0(x.reshape(-1, QK), mode="ref",
                                    mean_mult=MULT, theta=THETA)
    xh = Q.dequantize_row_ptq1_0(blob, x.size).reshape(x.shape)

    err = float(np.sqrt(((xh - x) ** 2).mean()) / np.sqrt((x ** 2).mean()))
    nz = float((xh != 0).mean()) * 100
    d_ratio = float((np.abs(x).reshape(-1, QK).mean(axis=1) * MULT).mean()
                    / np.abs(x).reshape(-1, QK).mean())
    allerr.append(err)
    print(f"{gguf_name:28s} {err * 100:9.2f}% {nz:7.2f}% {d_ratio:10.4f}")

print(f"\nmean reconstruction RMSE: {np.mean(allerr) * 100:.2f}%")
print(f"naive-ternary expectation (67% kept, Gaussian): ~46%")
print(f"\nnon-zero fraction should be 67.2% to match the reference.")
print("If RMSE is ~46%, the packer is a plain per-block ternary quantiser and the")
print("reference must be doing something structurally different to reach 4.18.")
