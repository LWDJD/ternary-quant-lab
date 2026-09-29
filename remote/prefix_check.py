"""Is the reference's kept set a prefix of my |x| ordering?

Sharper than an agreement rate.  If their criterion is any cutoff on my values,
their kept set must be exactly a prefix of my descending-|x| order (allowing for
the fact that a prefix has a fixed size, so the cut need not be at 86).  If some
blocks are prefixes and others are not, the exceptions are where to look.

Reports: how many blocks are exact prefixes, and among the rest, how far off they
are (symmetric difference size).
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

QK, FOLD = 128, 1024
POW3 = (1, 3, 9, 27, 81, 243)
REF = "/tmp/ref.gguf"
P = "model.language_model.layers.0."
JOBS = [("blk.0.ffn_gate.weight", P + "mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", P + "mlp.up_proj.weight")]
NROWS = 16


def decode_row(raw: bytes, nb: int) -> np.ndarray:
    out = np.empty((nb, QK), dtype=np.int8)
    for i in range(nb):
        blk = raw[i * 28:(i + 1) * 28]
        qs, qh = blk[0:24], blk[24:26]
        v = out[i]
        for c, eoff, boff in ((16, 0, 0), (8, 80, 16)):
            for n in range(5):
                base = eoff + n * c
                for m in range(c):
                    q = (int(qs[boff + m]) * POW3[n]) & 0xFF
                    v[base + m] = (((q * 3) >> 8) & 0xFF) - 1
        for n in range(4):
            for h in range(2):
                q = (int(qh[h]) * POW3[n]) & 0xFF
                v[120 + n * 2 + h] = (((q * 3) >> 8) & 0xFF) - 1
    return out


def load_rows(hf_name, rows):
    for shard in sorted(glob.glob("/tmp/models/qwen38-27b/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


r = GGUFReader(REF)
rmap = {t.name: t for t in r.tensors}
tot_blk = prefix_ok = 0
sizes = []
worst = []

for gguf_name, hf_name in JOBS:
    t = rmap.get(gguf_name)
    if t is None:
        continue
    raw = np.asarray(t.data).tobytes()
    rowbytes = (int(t.shape[0]) // QK) * 28
    nrows = min(NROWS, int(t.shape[1]))
    w = load_rows(hf_name, nrows)
    if w is None:
        continue
    width = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")
    x = PP.fold_weight(w, sign, FOLD).astype(np.float32)
    nb = width // QK

    for row in range(nrows):
        ref_t = decode_row(raw[row * rowbytes:(row + 1) * rowbytes], nb)
        mine = np.abs(x[row]).reshape(nb, QK)
        their_keep = ref_t != 0
        for b in range(nb):
            order = np.argsort(-mine[b])
            k = int(their_keep[b].sum())
            if k == 0 or k == QK:
                continue
            prefix = np.zeros(QK, dtype=bool)
            prefix[order[:k]] = True
            diff = int((prefix != their_keep[b]).sum())
            tot_blk += 1
            sizes.append(k)
            if diff == 0:
                prefix_ok += 1
            elif len(worst) < 200:
                worst.append(diff)
    print(f"  {gguf_name}: done")

print(f"\nblocks examined        {tot_blk}")
print(f"exact prefixes         {prefix_ok}  ({prefix_ok / max(tot_blk, 1) * 100:.2f}%)")
if sizes:
    print(f"kept per block         mean {np.mean(sizes):.2f}  min {min(sizes)}  max {max(sizes)}")
if worst:
    w = np.array(worst)
    print(f"symmetric diff, rest   mean {w.mean():.2f}  min {w.min()}  max {w.max()}")
print("\nNear 100% prefixes means their criterion is a cutoff on my values, so the")
print("residual lives in the values.  A low rate means it is not a cutoff at all.")
