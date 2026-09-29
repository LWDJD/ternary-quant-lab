"""Is my stored d off by a constant factor, or scattered?

Unsigned mean error was 2.86% (worst 4.6%).  A constant factor means the 1.40 in
d = 1.40*mean|x| is slightly wrong, and one number fixes everything.  Scatter means
some other variable is in play.

So report the signed ratio distribution: my d divided by the reference's.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

MINE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d27r2.gguf"
REF = "/tmp/ref.gguf"


def scales(t):
    raw = np.asarray(t.data).tobytes()
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    return np.frombuffer(np.ascontiguousarray(b[:, 26:28]).tobytes(),
                         dtype=np.float16).astype(np.float32)


mine = GGUFReader(MINE)
ref = GGUFReader(REF)
rmap = {t.name: t for t in ref.tensors}

ratios = []
per_tensor = []
for t in mine.tensors:
    if t.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    rt = rmap.get(t.name)
    if rt is None or rt.tensor_type != GGMLQuantizationType.PTQ1_0:
        continue
    a, b = scales(t), scales(rt)
    if a.shape != b.shape:
        continue
    ok = b > 0
    if not ok.any():
        continue
    r = a[ok] / b[ok]
    ratios.append(r)
    per_tensor.append((t.name, float(np.median(r))))
    if len(ratios) >= 400:
        break

allr = np.concatenate(ratios)
print(f"blocks compared: {allr.size}  tensors: {len(ratios)}")
print(f"mine/reference d:")
print(f"  median {np.median(allr):.5f}   mean {allr.mean():.5f}   "
      f"std {allr.std():.5f}")
for p in (1, 5, 25, 50, 75, 95, 99):
    print(f"  p{p:>2d} {np.percentile(allr, p):.5f}")
print(f"  min {allr.min():.5f}  max {allr.max():.5f}")
print(f"  within 0.5% of 1.0: {float(np.abs(allr - 1 < 0.005).mean()) * 100:.2f}%"
      if False else
      f"  |ratio-1| < 0.5%: {float((np.abs(allr - 1) < 0.005).mean()) * 100:.2f}%")

med = np.array([m for _, m in per_tensor])
print(f"\nper-tensor medians: mean {med.mean():.5f}  std {med.std():.5f} "
      f"min {med.min():.5f}  max {med.max():.5f}")
print("\nA tight cluster around one value means a single constant is wrong.")
print("A wide spread means another variable is involved.")
