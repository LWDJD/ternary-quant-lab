"""Is Bonsai's sign vector random, or derived?

The runtime builds the rotation itself from `block_size`, so the ONLY free
parameter in the transform is the per-width sign vector -- and the whitepaper
calls the transform "proprietary Caltech IP, a mathematically grounded framework
rather than a collection of heuristics".  If the signs are optimised (rather than
random), that is the missing piece.

Cheap structural tests first, on the reference artefact we already have.
"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")
from gguf import GGUFReader

REF = r"D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PTQ1_0.gguf"

r = GGUFReader(REF)
widths = [int(v) for v in r.fields["prism.hadamard.sign_widths"].contents()]
vals = np.array([int(v) for v in r.fields["prism.hadamard.sign_values"].contents()], dtype=np.int64)
print(f"widths = {widths}")
print(f"values len = {len(vals)} (sum of widths = {sum(widths)})")

off = 0
vecs = {}
for w in widths:
    v = vals[off:off + w]
    vecs[w] = v
    off += w
    ones = int((v == 1).sum())
    runs = int((v[1:] != v[:-1]).sum()) + 1
    print(f"\n--- width {w} ---")
    print(f"  +1: {ones}/{w}   ({ones / w:.4f})   -1: {w - ones}")
    print(f"  runs: {runs}  (random expectation ~{w / 2 + 1:.0f}, all-same = 1)")
    print(f"  first 40: {''.join('+' if x == 1 else '-' for x in v[:40])}")
    # autocorrelation at a few lags
    f = v.astype(np.float64)
    f = f - f.mean()
    denom = (f * f).sum()
    if denom > 0:
        ac = [round(float((f[:-k] * f[k:]).sum() / denom), 4) for k in (1, 2, 3, 4, 8, 16)]
        print(f"  autocorr lags 1,2,3,4,8,16: {ac}")
    # is it a prefix / hash slice of something simple?
    for seedname in ("0", "1", "42", "1337", str(w)):
        h = hashlib.sha256(seedname.encode()).digest()
        bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:w] if w <= 256 else None
        if bits is not None:
            cand = np.where(bits == 1, 1, -1)
            if np.array_equal(cand, v):
                print(f"  MATCHES sha256({seedname}) bits!")
    # periodicity: does it repeat with period p?
    for p in (2, 4, 8, 16, 32, 64, 128):
        if p < w:
            rep = np.tile(v[:p], w // p + 1)[:w]
            if np.array_equal(rep, v):
                print(f"  REPEATS with period {p}")

print("\n--- relations between widths ---")
ks = sorted(vecs)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = vecs[ks[i]], vecs[ks[j]]
        n = min(len(a), len(b))
        same = int((a[:n] == b[:n]).sum())
        # also compare b's head against a coarse view (pairwise / strided)
        print(f"  {ks[i]} vs {ks[j]}: first {n} agree {same}/{n} ({same / n:.3f})")
        for stride in (2, 4, 8):
            if len(a) >= stride:
                if np.array_equal(np.repeat(a[: len(b) // stride], stride)[: len(b)], b[: (len(b) // stride) * stride]):
                    print(f"     {ks[j]} == repeat({ks[i]}, {stride})")
