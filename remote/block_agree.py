"""Per-tensor check of my pack against the reference, at block granularity.

The 26 trit bytes of a PTQ1_0 block are a bijection of that block's 128 trits, so
"are these 26 bytes equal" IS "are these 128 trits equal" -- no decoding needed.
Vectorised, so the whole model takes seconds instead of hours.

What this is for: a rule that is right everywhere still shows ~10% of blocks
differing (boundary flips).  A tensor that was folded along the wrong axis, or
quantised when it should have been protected, shows up as almost no equal blocks.
That is the profile worth hunting.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

MINE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d27.gguf"
REF = "/tmp/bonsai2-ptq10.gguf"
BLK = 28


def blocks(t) -> np.ndarray:
    raw = np.asarray(t.data).tobytes()
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLK)


mine = GGUFReader(MINE)
ref = GGUFReader(REF)
rmap = {t.name: t for t in ref.tensors}
print(f"mine {len(mine.tensors)} tensors   ref {len(ref.tensors)} tensors\n")

rows = []
for t in mine.tensors:
    if t.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    rt = rmap.get(t.name)
    if rt is None or rt.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    a, b = blocks(t), blocks(rt)
    if a.shape != b.shape:
        print(f"  SHAPE {t.name}: {a.shape} vs {b.shape}")
        continue
    # Block identity is useless here: at 90% trit agreement a 128-trit block
    # matches exactly with probability 0.9**128 ~ 1e-6.  Bytewise agreement over
    # the 26 trit bytes is the informative number: with 5 trits per byte it sits
    # near 0.9**5 = 0.59, and a tensor folded along a wrong axis collapses to ~1/256.
    bytewise = float((a[:, :26] == b[:, :26]).mean()) * 100
    rows.append((t.name, bytewise, a.shape[0]))


rows.sort(key=lambda r: r[1])
print(f"{'tensor':42s} {'bytewise identical':>18s} {'blocks':>9s}")
for name, pct, nb in rows[:14]:
    print(f"{name:42s} {pct:16.2f}% {nb:9d}")
print("   ...")
for name, pct, nb in rows[-4:]:
    print(f"{name:42s} {pct:16.2f}% {nb:9d}")

tot = sum(r[2] for r in rows)
acc = sum(r[1] / 100 * r[2] for r in rows) / max(tot, 1)
print(f"\ntensors compared {len(rows)}   blocks {tot}")
print(f"weighted bytewise rate: {acc * 100:.2f}%   (chance 0.39%, 0.9**5 = 59%)")
for lo in (50, 20, 5, 1):
    bad = [r for r in rows if r[1] < lo]
    print(f"  below {lo:2d}%: {len(bad):3d} tensors")
bad = [r for r in rows if r[1] < 50]
if bad:
    print("\nworst tensors:")
    for name, pct, nb in bad[:15]:
        print(f"   {name:40s} {pct:6.2f}%  ({nb} blocks)")
