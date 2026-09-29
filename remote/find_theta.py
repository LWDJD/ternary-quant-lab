"""Pin the reference's decision threshold.

Their per-block scale is d ~= 1.40 * mean|x|, but round-to-nearest at 0.5*d only
reproduces 57.8% non-zero while the reference holds 67.2%.  So the boundary sits
below 0.5*d.  Given their own d, sweep theta in

    trit = 0 if |x| < theta*d else sign(x)

and report trit agreement against their stored trits.  No packing, no PPL: this
isolates the decision rule from the scale rule.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

QK = 128           # quantisation block (PTQ1_0)
FOLD = 1024        # rotation block (prism.hadamard.block_size) -- NOT the quant block
HF = "/tmp/q38-27b"
SMALL = "/mnt/workspace/prismwork/ref_small.npz"
P = "model.language_model.layers.0."
HFNAMES = {
    "blk_0_ffn_gate_weight": P + "mlp.gate_proj.weight",
    "blk_0_ffn_up_weight": P + "mlp.up_proj.weight",
    "blk_0_ssm_out_weight": P + "linear_attn.out_proj.weight",
}


def load_rows(hf_name: str, rows: int):
    for shard in sorted(glob.glob(f"{HF}/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


s = np.load(SMALL)
print(f"keys: {sorted(s.files)}\n")

THETAS = [0.16, 0.20, 0.24, 0.27, 0.30, 0.33, 0.36, 0.38, 0.40, 0.45, 0.50]
rows_total = 0
agree_sum = np.zeros(len(THETAS))

print(f"{'tensor':24s} " + " ".join(f"{t:6.3f}" for t in THETAS))
nz_sum = np.zeros(len(THETAS))
for base, hf_name in HFNAMES.items():
    tk, dk = base + "__t", base + "__d"
    if tk not in s:
        print(f"  {base}: not in sample")
        continue
    tgt, dref = s[tk], s[dk].astype(np.float32)
    rows, cols = tgt.shape
    w = load_rows(hf_name, rows)
    if w is None or w.shape[1] % QK:
        print(f"  {base}: HF missing or bad width")
        continue
    full = w.shape[1]
    if cols > full:
        print(f"  {base}: sample width {cols} exceeds HF width {full}")
        continue

    # fold at full width (the sign vector spans the whole input dim), then keep the
    # same corner the sample covers
    sign = np.load(f"/mnt/workspace/prismwork/sign_{full}.npy")
    fd = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :cols]

    line = []
    nb = cols // QK
    d_rep = np.repeat(dref.reshape(rows, nb), QK, axis=1)
    for i, th in enumerate(THETAS):
        pred = np.where(np.abs(fd) >= th * d_rep, np.sign(fd), 0).astype(np.int8)
        agree = float((pred == tgt).mean())
        line.append(agree)
        agree_sum[i] += agree * tgt.size
        nz_sum[i] += float((pred != 0).mean()) * tgt.size
    rows_total += tgt.size
    nz = float((tgt != 0).mean()) * 100
    print(f"{base:24s} " + " ".join(f"{a * 100:6.2f}" for a in line)
          + f"   [ref nonzero {nz:.2f}%]")

print("\n" + "=" * 70)
print(f"weighted over {rows_total} samples   (reference non-zero = 67.2%)")
print(f"{'theta':>6s}  {'agree':>7s}  {'nonzero':>8s}   bar")
for i, t in enumerate(THETAS):
    acc = agree_sum[i] / max(rows_total, 1)
    nz = nz_sum[i] / max(rows_total, 1)
    print(f"{t:6.3f}  {acc * 100:6.2f}%  {nz * 100:7.2f}%   " + "#" * int(round(acc * 50)))

best = int(np.argmax(agree_sum))
print(f"\nbest agreement  theta={THETAS[best]:.3f}  -> "
      f"{agree_sum[best] / max(rows_total, 1) * 100:.2f}%")
nz_best = int(np.argmin(np.abs(nz_sum / max(rows_total, 1) - 0.672)))
print(f"closest to 67.2% non-zero  theta={THETAS[nz_best]:.3f}  -> "
      f"{nz_sum[nz_best] / max(rows_total, 1) * 100:.2f}% nonzero, "
      f"agreement {agree_sum[nz_best] / max(rows_total, 1) * 100:.2f}%")
print("chance is 33.3%. If one theta tops both columns, that is the rule.")
