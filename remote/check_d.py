"""What is the reference's per-block d, relative to the folded weight statistics?

The statistic that separates the two hypotheses:
   d / max|x|    -> 12.7% cv   (a max-driven rule)
   d / mean|x|   ->  2.0% cv   (a mean-driven rule)

Measured on the *folded* weight, because the Hadamard rotation is what the
reference quantises, and it changes the distribution the rule sees.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

QK = 128
HF = "/tmp/q38-27b"
SAMPLE = "/mnt/workspace/prismwork/ref_d_sample.npz"
P = "model.language_model.layers.0."
HFNAMES = {
    "blk.0.ffn_gate.weight": P + "mlp.gate_proj.weight",
    "blk.0.ffn_up.weight": P + "mlp.up_proj.weight",
    "blk.0.ffn_down.weight": P + "mlp.down_proj.weight",
    "blk.0.ssm_out.weight": P + "linear_attn.out_proj.weight",
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


s = np.load(SAMPLE)

# sign vectors may arrive bit-packed (they are +/-1, so one bit each)
import os  # noqa: E402
pk = "/mnt/workspace/prismwork/sign_packed.npz"
if os.path.exists(pk):
    for key in np.load(pk).files:
        width = int(key.split("_")[1])
        bits = np.load(pk)[key]
        vec = np.unpackbits(bits)[:width].astype(np.float32) * 2.0 - 1.0
        np.save(f"/mnt/workspace/prismwork/sign_{width}.npy", vec)
        print(f"unpacked sign_{width}: n={vec.size} pos={float((vec > 0).mean()) * 100:.1f}%")

print(f"sample keys: {sorted(s.files)}\n")
print(f"{'tensor':24s} {'n':>5s} {'d/mean|x|':>10s} {'cv%':>6s} "
      f"{'d/max|x|':>10s} {'cv%':>6s}")

for name, hf_name in HFNAMES.items():
    key = name.replace(".", "_") + "__d"
    if key not in s:
        print(f"  {name}: not in sample")
        continue
    dref = s[key].astype(np.float32)
    rows, nblk = dref.shape
    w = load_rows(hf_name, rows)
    if w is None:
        print(f"  {name}: HF tensor missing ({hf_name})")
        continue
    if w.shape[1] % QK:
        print(f"  {name}: input dim {w.shape[1]} not a multiple of {QK}")
        continue

    sign = np.load("/mnt/workspace/prismwork/sign_%d.npy" % w.shape[1]) \
        if glob.glob("/mnt/workspace/prismwork/sign_%d.npy" % w.shape[1]) \
        else np.ones(w.shape[1], np.float32)
    fd = PP.fold_weight(w, sign, QK).astype(np.float32)

    # per-block statistics of the folded weight
    rb = fd.reshape(rows, nblk, QK)
    mean_abs = np.abs(rb).mean(axis=2)
    max_abs = np.abs(rb).max(axis=2)

    r_mean = dref / np.maximum(mean_abs, 1e-30)
    r_max = dref / np.maximum(max_abs, 1e-30)
    print(f"{name:24s} {rows * nblk:5d} {r_mean.mean():10.4f} "
          f"{r_mean.std() / r_mean.mean() * 100:6.2f} {r_max.mean():10.4f} "
          f"{r_max.std() / r_max.mean() * 100:6.2f}")

print("\nA stable d/mean|x| (low cv) with a matching constant is the mean-driven rule;")
print("d/max|x| varying a lot is what a max-driven rule looks like.")
print(f"\nreference non-zero fraction: 67.2% -> mean|t| = 0.672 -> d = 1.488*mean|x|")
print("if the scale were set by matching mean|d*t| to mean|x|.")
