"""Sweep the value grid: which mantissa width reproduces the reference's excess?

The reference keeps 86 per block as a hard floor, exceeded in 3.1% of blocks with
mean excess 0.037.  "Keep everything >= the 86th largest" turns that excess into a
tie count, so the excess rate is a function of the value grid: finer grid, fewer
ties.  fp16 (10 mantissa bits) gives 1.2%, bf16 (7 stored bits) gives 8.6%, so the
answer should sit between.

This rounds the FOLDED values to a configurable mantissa width and reports the
excess rate and mean, looking for a simultaneous match on both.
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
NROWS = 8


def round_mbits(a: np.ndarray, mb: int) -> np.ndarray:
    """Round a float32 to `mb` mantissa bits (23 = unchanged)."""
    if mb >= 23:
        return a
    shift = 23 - mb
    add = np.uint32(1 << (shift - 1))
    mask = np.uint32(0xFFFFFFFF << shift)
    u = a.view(np.uint32)
    r = ((u + add) & mask).view(np.float32)
    return np.where(np.isfinite(r), r, a)


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
ref_ex, folds = [], []
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
        for b in range(nb):
            c = int((ref_t[b] != 0).sum())
            if c:
                ref_ex.append(c - K)
        folds.append(x[row].reshape(nb, QK))
    print(f"  {gguf_name}: done")

ref_ex = np.array(ref_ex)
print(f"\nreference: rate {float((ref_ex > 0).mean()) * 100:.2f}%  "
      f"mean {ref_ex.mean():.4f}  n={ref_ex.size}")

print(f"\n{'mantissa':>8s} {'tie rate':>10s} {'mean excess':>12s}")
for mb in (8, 9, 10, 11, 12, 13, 16, 23):
    ex = []
    for blk in folds:
        v = np.abs(round_mbits(blk, mb))
        s = np.sort(v, axis=1)[:, ::-1]
        cut = s[:, K - 1]
        ex.append((v >= cut[:, None]).sum(axis=1) - K)
    ex = np.concatenate(ex)
    mark = ""
    if abs(float((ex > 0).mean()) * 100 - 3.1) < 1.0:
        mark = "   <- rate matches"
    print(f"{mb:8d} {float((ex > 0).mean()) * 100:9.2f}% {ex.mean():12.4f}{mark}")

print("\nreference target: tie rate 3.10%, mean excess 0.037")
