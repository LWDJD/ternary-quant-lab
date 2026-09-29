"""How big is the gap at the rank-86 boundary, relative to d?

If the reference's decision is a rank rule on |x|, then the elements it can get
"wrong" relative to a slightly different fold are those whose gap to the boundary
is on the order of the fold difference.  So this number sets how precisely the
fold would have to match: gap/d << float32 error means precision is not a route.

Reported in units of d, plus the implied required relative precision.
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
K = 86
HF = "/tmp/models/qwen38-27b"
P = "model.language_model.layers.0."
TENSORS = [P + "mlp.gate_proj.weight", P + "mlp.up_proj.weight",
           P + "mlp.down_proj.weight"]


def load(hf_name, shard_hint=1):
    for shard in sorted(glob.glob(f"{HF}/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:64].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


gaps = []
for hf_name in TENSORS:
    w = load(hf_name)
    if w is None:
        print(f"  {hf_name}: missing")
        continue
    width = w.shape[1]
    try:
        sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")
    except Exception:  # noqa: BLE001
        print(f"  {hf_name}: no sign for width {width}")
        continue
    x = PP.fold_weight(w, sign, FOLD).astype(np.float32)
    nblk = width // QK
    rb = np.abs(x.reshape(w.shape[0], nblk, QK))
    s = np.sort(rb, axis=2)[:, :, ::-1]           # descending
    d = np.float32(1.40) * np.abs(x.reshape(w.shape[0], nblk, QK)).mean(axis=2)
    gap = (s[:, :, K - 1] - s[:, :, K]) / d        # 86th minus 87th, in units of d
    gaps.append(gap.ravel())
    print(f"  {hf_name:40s} blocks={gap.size:6d}  "
          f"median gap/d = {np.median(gap) * 100:.3f}%  "
          f"mean {gap.mean() * 100:.3f}%")

g = np.concatenate(gaps)
if len(g):
    print(f"\nall: n={g.size}  median {np.median(g) * 100:.3f}%  "
          f"p10 {np.percentile(g, 10) * 100:.3f}%  "
          f"p90 {np.percentile(g, 90) * 100:.3f}%")
    print(f"float32 relative resolution is ~6e-8, i.e. 6e-6 %")
    print(f"ratio: the boundary gap is {np.median(g) / 6e-8:.0f}x the float32 step")
print("\nIf the gap is orders of magnitude above the fp32 step, fold precision")
print("cannot be what decides the boundary: their criterion differs in kind.")
