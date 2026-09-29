"""Are the disagreements near-ties, or structural?

If the reference's criterion were the same as mine (top-86 by |x|) on slightly
different values, every disagreement must sit at my rank ~86: those are the only
elements a small perturbation can move across the boundary.  If instead their
criterion differs in kind, disagreements will spread over a wide band of ranks,
and may cluster at particular positions inside the block.

So: for each disagreeing element, record its rank in my ordering, and its offset
inside the 128-block.  Two histograms settle the question.
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

QK, FOLD, K = 128, 1024, 86
POW3 = (1, 3, 9, 27, 81, 243)
REF = "/tmp/ref.gguf"
P = "model.language_model.layers.0."
JOBS = [("blk.0.ffn_gate.weight", P + "mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", P + "mlp.up_proj.weight")]
NBLK = 400


def decode_blocks(raw: bytes, nblk: int) -> np.ndarray:
    out = np.empty((nblk, QK), dtype=np.int8)
    for i in range(nblk):
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
rank_hist = np.zeros(QK, dtype=np.int64)
pos_hist = np.zeros(QK, dtype=np.int64)
n_flip = n_tot = 0

for gguf_name, hf_name in JOBS:
    t = rmap.get(gguf_name)
    if t is None:
        print(f"  {gguf_name}: not in reference")
        continue
    raw = np.asarray(t.data).tobytes()
    rowbytes = (int(t.shape[0]) // QK) * 28
    # take whole rows so the element order matches the weight layout
    nrows = min(4, int(t.shape[1]))
    w = load_rows(hf_name, nrows)
    if w is None:
        print(f"  {hf_name}: missing")
        continue
    width = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")
    x = PP.fold_weight(w, sign, FOLD).astype(np.float32)[:, :width]

    for row in range(nrows):
        nb = min(NBLK // nrows, width // QK)
        raw_row = raw[row * rowbytes:row * rowbytes + nb * QK // QK * 28]
        ref_t = decode_blocks(raw_row, nb)                       # (nb, QK)
        mine = x[row, :nb * QK].reshape(nb, QK)
        order = np.argsort(-np.abs(mine), axis=1)                # descending |x|
        rank_of = np.empty_like(order)
        np.put_along_axis(rank_of, order, np.arange(QK)[None, :].repeat(nb, 0), axis=1)
        my_t = np.zeros_like(ref_t)
        np.put_along_axis(my_t, order[:, :K], np.take_along_axis(np.sign(mine), order[:, :K], axis=1), axis=1)

        diff = my_t != ref_t
        n_flip += int(diff.sum())
        n_tot += diff.size
        for b, p in zip(*np.nonzero(diff)):
            pos_hist[p] += 1
            rank_hist[rank_of[b, p]] += 1
    print(f"  {gguf_name}: done")

print(f"\ndisagreements {n_flip}/{n_tot} ({n_flip / n_tot * 100:.2f}%)")
print("\nrank of disagreeing element in my ordering (0 = largest):")
for lo in range(0, QK, 16):
    band = rank_hist[lo:lo + 16].sum()
    frac = band / max(rank_hist.sum(), 1) * 100
    print(f"  rank {lo:3d}-{lo + 15:3d}: {band:7d}  {frac:5.1f}%  {'#' * int(frac)}")
top = rank_hist[76:97].sum() / max(rank_hist.sum(), 1) * 100
print(f"\nshare at rank 76-96 (around the boundary): {top:.1f}%")

print("\noffset inside the 128-block (structural clustering would show up here):")
half = [pos_hist[:64].sum(), pos_hist[64:].sum()]
print(f"  first half {half[0]}  second half {half[1]}")
peak = int(np.argmax(pos_hist))
print(f"  busiest offset {peak} with {pos_hist[peak]} (uniform would be ~{pos_hist.sum() / QK:.0f})")
print("\nAll flips at rank ~86 means near-ties (value difference).")
print("A broad rank spread means their criterion is not a rank on |x|.")
