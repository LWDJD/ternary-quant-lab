"""Diagnose the chance-level agreement: is d wrong, or is the fold wrong?

Three checks:
  1. If d is correct, |fd| over the reference's non-zero positions should sit at a
     plausible fraction of d (roughly 0.5-0.9), and |fd| at zero positions should
     mostly be below d.
  2. Correlation between |fd| and the reference's implied |d*t|.
  3. Agreement for several fold conventions (sign before/after the Hadamard, sign
     vector as-is or reversed), to see whether the convention is simply different.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

QK = 128          # quantisation block (PTQ1_0)
FOLD = 1024       # rotation block (prism.hadamard.block_size) -- NOT the quant block
HF = "/tmp/q38-27b"
P = "model.language_model.layers.0."
HFNAMES = {"blk_0_ffn_gate_weight": P + "mlp.gate_proj.weight",
           "blk_0_ffn_up_weight": P + "mlp.up_proj.weight"}


def load_rows(hf_name, rows):
    for shard in sorted(glob.glob(f"{HF}/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


def hadamard(n, block, dtype=np.float32):
    """Normalized Walsh-Hadamard. H[i][j]=(-1)^popcount(i&j), natural (bit-reversed) order."""
    idx = np.arange(block)
    and_ = idx[:, None] & idx[None, :]
    H = np.array([[bin(int(v)).count("1") & 1 for v in row] for row in and_], dtype=dtype)
    H = 1.0 - 2.0 * H
    return H / np.sqrt(block)


def hadamard_seq(block, dtype=np.float32):
    """Same, but sequency-ordered: index i is read bit-reversed."""
    bits = int(np.log2(block))
    idx = np.arange(block)
    rev = np.array([int(format(i, f"0{bits}b")[::-1], 2) for i in idx])
    and_ = rev[:, None] & rev[None, :]
    H = np.array([[bin(int(v)).count("1") & 1 for v in row] for row in and_], dtype=dtype)
    return (1.0 - 2.0 * H) / np.sqrt(block)


s = np.load("/mnt/workspace/prismwork/ref_small.npz")
H = hadamard(0, FOLD)
HS = hadamard_seq(FOLD)
print(f"fold block = {FOLD} (rotation), quant block = {QK}")

for base, hf_name in HFNAMES.items():
    tgt = s[base + "__t"]
    dref = s[base + "__d"].astype(np.float32)
    rows, cols = tgt.shape
    nb = cols // QK
    w = load_rows(hf_name, rows)
    full = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{full}.npy")

    print(f"\n=== {base} ===")

    # candidates for the folded weight
    variants = {}
    variants["PP.fold_weight"] = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :cols]
    ws = w * sign[None, :]
    variants["sign then H"] = (ws.reshape(rows, full // FOLD, FOLD) @ H
                               ).reshape(rows, full).astype(np.float32)[:, :cols]
    variants["sign then Hseq"] = (ws.reshape(rows, full // FOLD, FOLD) @ HS
                                  ).reshape(rows, full).astype(np.float32)[:, :cols]
    rb2 = w.reshape(rows, full // FOLD, FOLD)
    variants["H then sign"] = ((rb2 @ H).reshape(rows, full)
                               * sign[None, :]).astype(np.float32)[:, :cols]
    variants["sign reversed"] = ((w * sign[::-1][None, :]).reshape(rows, full // FOLD, FOLD) @ H
                                 ).reshape(rows, full).astype(np.float32)[:, :cols]

    d_rep = np.repeat(dref, QK, axis=1)
    implied = np.abs(d_rep * tgt)

    for label, fd in variants.items():
        nz_mean = float(np.abs(fd)[tgt != 0].mean())
        z_mean = float(np.abs(fd)[tgt == 0].mean())
        agree = float((np.where(np.abs(fd) >= 0.5 * d_rep, np.sign(fd), 0) == tgt).mean())
        corr = float(np.corrcoef(np.abs(fd).ravel(), implied.ravel())[0, 1])
        print(f"  {label:16s} <|x|>nz/d={nz_mean / dref.mean():5.3f} "
              f"<|x|>zero/d={z_mean / dref.mean():5.3f} "
              f"corr(|x|,|d*t|)={corr:6.3f}  agree@0.5={agree * 100:5.2f}%")

print("\nIf <|x|>nz/d ~ 0.7 and <|x|>zero/d ~ 0.2, d and the fold are both right.")
print("A flat or inverted pattern means the fold places values in the wrong slots.")
