"""Trit agreement between my pack and the reference, over EVERY shared tensor.

A 4-row sample showed the rule and the fold are right; this checks the whole
model, which is where a per-tensor exception (a kind that should not have been
quantised, or a width whose sign vector is shared wrongly) would show up as a
handful of badly-wrong tensors among otherwise perfect ones.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

MINE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d27.gguf"
REF = "/tmp/bonsai2-ptq10.gguf"
QK = 128
POW3 = (1, 3, 9, 27, 81, 243)


def decode_tensor(t) -> np.ndarray:
    """Whole tensor -> int8 trits, shape (ne1, ne0)."""
    ne0, ne1 = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(t.data).tobytes()
    nb = ne0 // QK
    rowbytes = nb * 28
    out = np.empty((ne1, ne0), dtype=np.int8)
    tpl = np.zeros(QK, dtype=np.int8)
    for i in range(ne1):
        row = raw[i * rowbytes:(i + 1) * rowbytes]
        for b in range(nb):
            blk = row[b * 28:(b + 1) * 28]
            qs, qh = blk[0:24], blk[24:26]
            v = tpl
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
            out[i, b * QK:(b + 1) * QK] = v
    return out


mine = GGUFReader(MINE)
ref = GGUFReader(REF)
rmap = {t.name: t for t in ref.tensors}
print(f"mine: {len(mine.tensors)} tensors   ref: {len(ref.tensors)}\n")

rows = []
for t in mine.tensors:
    if t.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    rt = rmap.get(t.name)
    if rt is None or rt.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    a = decode_tensor(t)
    b = decode_tensor(rt)
    if a.shape != b.shape:
        print(f"  SHAPE {t.name}: {a.shape} vs {b.shape}")
        continue
    agree = float((a == b).mean()) * 100
    nz = float((b != 0).mean()) * 100
    rows.append((t.name, agree, nz, a.size))

rows.sort(key=lambda r: r[1])
print(f"{'tensor':40s} {'agree':>7s} {'refNonzero':>11s} {'n':>10s}")
for name, agree, nz, n in rows[:12]:
    print(f"{name:40s} {agree:6.2f}% {nz:10.2f}% {n:10d}")
print("   ...")
for name, agree, nz, n in rows[-5:]:
    print(f"{name:40s} {agree:6.2f}% {nz:10.2f}% {n:10d}")

tot = sum(r[3] for r in rows)
acc = sum(r[1] / 100 * r[3] for r in rows) / max(tot, 1)
print(f"\ntensors compared: {len(rows)}  elements: {tot}")
print(f"weighted agreement: {acc * 100:.2f}%")
bad = [r for r in rows if r[1] < 60]
print(f"tensors below 60%: {len(bad)}")
for name, agree, nz, n in bad[:15]:
    print(f"   {name:38s} {agree:6.2f}%   refNonzero {nz:.2f}%")
