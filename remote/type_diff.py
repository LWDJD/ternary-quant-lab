"""Which tensors are ternary in my pack vs the reference?

The no-quant control settles it: rotated F16 scores PPL 3.46 against the
reference's 4.18, so the fold, the sign vectors and the runtime handling are all
correct and the entire 370x blow-up lives in the quantisation step.  Ternary costs
the reference only 4.18 everywhere, so my quantisation is doing something
structurally wrong, not merely inexact.

The remaining structural candidate, raised long ago and never checked
systematically: quantising a tensor that should have been left alone.  So list the
type of every tensor in both files and diff the sets.
"""
from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

MINE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d27r2.gguf"
REF = "/tmp/ref.gguf"

a = GGUFReader(MINE)
b = GGUFReader(REF)
amap = {t.name: str(t.tensor_type).split(".")[-1] for t in a.tensors}
bmap = {t.name: str(t.tensor_type).split(".")[-1] for t in b.tensors}

print("mine:", dict(Counter(amap.values())))
print("ref :", dict(Counter(bmap.values())))

only_mine = sorted(set(amap) - set(bmap))
only_ref = sorted(set(bmap) - set(amap))
print(f"\nonly in mine: {len(only_mine)}")
for n in only_mine[:12]:
    print(f"   {n}  ({amap[n]})")
print(f"only in ref: {len(only_ref)}")
for n in only_ref[:12]:
    print(f"   {n}  ({bmap[n]})")

diffs = [(n, amap[n], bmap[n]) for n in sorted(set(amap) & set(bmap))
         if amap[n] != bmap[n]]
print(f"\ntype mismatches: {len(diffs)}")
for n, x, y in diffs[:40]:
    print(f"   {n:44s} mine {x:8s} ref {y}")
