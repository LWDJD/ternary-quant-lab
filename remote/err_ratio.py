"""How much worse is my trit decision than the reference's, in weight space?

Uses the same per-block d for both sides, so this isolates the DECISION difference
and reports it as a reconstruction error ratio.  If the ratio is modest (say <1.3x)
then the ~10% trit mismatch cannot explain a catastrophic model failure, and the
search should move elsewhere.
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
HF = "/tmp/models/qwen38-27b"
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


s = np.load("/mnt/workspace/prismwork/ref_small.npz")
e_ref = e_mine = n = 0.0
flip = 0
for base, hf_name in HFNAMES.items():
    tgt, dref = s[base + "__t"], s[base + "__d"].astype(np.float32)
    rows, cols = tgt.shape
    nb = cols // QK
    w = load_rows(hf_name, rows)
    if w is None:
        print(f"  {base}: weights missing")
        continue
    sign = np.load(f"/mnt/workspace/prismwork/sign_{w.shape[1]}.npy")
    x = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :cols]

    rb = x.reshape(rows, nb, QK)
    d = dref.reshape(rows, nb)
    t_ref = tgt.reshape(rows, nb, QK).astype(np.float32)
    # my rule: keep the 86 largest magnitudes per block
    idx = np.argsort(-np.abs(rb), axis=2)[:, :, :86]
    t_mine = np.zeros_like(rb)
    np.put_along_axis(t_mine, idx, np.take_along_axis(np.sign(rb), idx, axis=2), axis=2)

    e_ref += float(((d[:, :, None] * t_ref - rb) ** 2).sum())
    e_mine += float(((d[:, :, None] * t_mine - rb) ** 2).sum())
    flip += int((t_mine != t_ref).sum())
    n += rb.size

print(f"elements {int(n)}   trits differing {flip} ({flip / n * 100:.2f}%)")
print(f"MSE with the reference's trits : {e_ref / n:.6e}")
print(f"MSE with my trits              : {e_mine / n:.6e}")
print(f"ratio mine/reference           : {e_mine / e_ref:.4f}x")
print("\nA ratio near 1.1-1.2 means the decision difference is minor in weight space")
print("and cannot by itself explain a model that outputs nothing but repetition.")
