"""Where do the remaining ~10% of trit disagreements live?

If the fold is right and only the decision boundary is imperfect, disagreements
should pile up near |x| ~ theta*d and vanish elsewhere.  If the fold places values
in the wrong slots, disagreement should be flat across the range.

Also reports sign-only agreement (ignoring zeros) and the zero-pattern agreement,
which separates "wrong slot" from "wrong side of the threshold".
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
THETA = 0.38


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
BINS = [(0.0, 0.2), (0.2, 0.30), (0.30, 0.34), (0.34, 0.42), (0.42, 0.5),
        (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 99.0)]
tally = np.zeros((len(BINS), 2))
sign_hit = sign_tot = zero_hit = zero_tot = 0

for base, hf_name in HFNAMES.items():
    tgt, dref = s[base + "__t"], s[base + "__d"].astype(np.float32)
    rows, cols = tgt.shape
    nb = cols // QK
    w = load_rows(hf_name, rows)
    full = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{full}.npy")
    fd = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :cols]
    d_rep = np.repeat(dref, QK, axis=1)
    ratio = np.abs(fd) / d_rep
    pred = np.where(np.abs(fd) >= THETA * d_rep, np.sign(fd), 0).astype(np.int8)

    for i, (lo, hi) in enumerate(BINS):
        m = (ratio >= lo) & (ratio < hi)
        if m.sum() == 0:
            continue
        tally[i, 0] += int((pred[m] == tgt[m]).sum())
        tally[i, 1] += int(m.sum())

    nz = tgt != 0
    sign_hit += int((np.sign(fd)[nz] == tgt[nz]).sum())
    sign_tot += int(nz.sum())
    z = ~nz
    zero_hit += int((pred[z] == 0).sum())
    zero_tot += int(z.sum())

print(f"theta={THETA}, fold block={FOLD}, quant block={QK}\n")
print(f"{'|x|/d range':>16s} {'n':>8s} {'agreement':>10s}")
for (lo, hi), (hit, tot) in zip(BINS, tally):
    if tot == 0:
        continue
    print(f"{f'{lo:.2f}-{hi:.2f}':>16s} {int(tot):8d} {hit / tot * 100:9.2f}%")

print(f"\nsign agreement (over reference non-zero slots): "
      f"{sign_hit / max(sign_tot, 1) * 100:.2f}%")
print(f"zero-pattern agreement (over reference zero slots): "
      f"{zero_hit / max(zero_tot, 1) * 100:.2f}%")
print("\nEdge pile-up means the fold is fine and only the boundary flips values.")
print("A flat profile means values sit in the wrong slots.")
