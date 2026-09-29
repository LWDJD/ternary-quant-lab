"""Compare the stored per-block scale d between my pack and the reference.

Trit agreement says nothing about magnitude: a trit of +1 reconstructs as +d, so a
wrong d makes every non-zero weight the wrong size while leaving the trit pattern
untouched.  That would explain a 309x perplexity gap alongside 90% trit agreement.

d lives in the last two bytes of each 28-byte block.  Both files are GGUFs, and my
pack used the reference's own sign vectors, so no decoder is needed: compare bytes
26:28 directly, block by block.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

MINE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d27r2.gguf"
REF = "/tmp/ref.gguf"


def scales(t) -> np.ndarray:
    raw = np.asarray(t.data).tobytes()
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    return np.frombuffer(np.ascontiguousarray(b[:, 26:28]).tobytes(),
                         dtype=np.float16).astype(np.float32)


mine = GGUFReader(MINE)
ref = GGUFReader(REF)
rmap = {t.name: t for t in ref.tensors}

rows = []
for t in mine.tensors:
    if t.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    rt = rmap.get(t.name)
    if rt is None or rt.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    a, b = scales(t), scales(rt)
    if a.shape != b.shape:
        continue
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-30)
    rows.append((t.name, float(rel.mean()) * 100, float(np.median(rel)) * 100,
                 float((a == b).mean()) * 100))
    if len(rows) >= 400:
        break

rows.sort(key=lambda r: -r[1])
print(f"{'tensor':40s} {'mean rel err':>13s} {'median':>8s} {'exact':>8s}")
for r in rows[:10]:
    print(f"{r[0]:40s} {r[1]:12.4f}% {r[2]:7.4f}% {r[3]:7.2f}%")
print("   ...")
for r in rows[-4:]:
    print(f"{r[0]:40s} {r[1]:12.4f}% {r[2]:7.4f}% {r[3]:7.2f}%")

m = np.array([r[1] for r in rows])
ex = np.array([r[3] for r in rows])
print(f"\ntensors {len(rows)}")
print(f"mean relative d error   mean {m.mean():.4f}%   worst {m.max():.4f}%")
print(f"exactly equal d bytes   mean {ex.mean():.2f}%")
print("\nA few percent of magnitude error is enough to ruin a model while leaving")
print("trit agreement untouched.")
