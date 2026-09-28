"""Verify the scale-matching rule against least squares, on real folded weights.

Rule under test (inferred from the reference's own trit statistics):
    choose d such that  mean(|d*t|) == mean(|x|),  i.e.  d = mean|x| / mean|t|
solved by fixed-point iteration, NOT by minimising squared error.

Reference anchors: Ternary-Bonsai-2-27B has 67.2% non-zero on every tensor, so
mean|t| = 0.672 there, giving d = 1.488 * mean|x|.  Our least-squares fit lands at
~50.8% non-zero -- i.e. it zeroes far more weights, and on the reference's own d the
reconstruction would only show 58% non-zero.
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-pack")
sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")

import pack_gguf as P  # noqa: E402
import prism_pack as PP  # noqa: E402
import ptq1_0 as Q  # noqa: E402
from gguf import GGUFReader  # noqa: E402

BLOCK = 128
GGUF = r"D:\AI\lmstudio\models\lmstudio-community\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-Q3_K_L.gguf"


def rule_ls(x, iters=3):
    """current: minimise squared error."""
    d = np.abs(x).max(axis=1).astype(np.float32)
    t = np.zeros_like(x)
    for _ in range(iters):
        t = np.clip(np.rint(x / d[:, None]), -1, 1)
        num = (x * t).sum(axis=1)
        den = (t * t).sum(axis=1)
        d = np.where(den > 0, np.abs(num) / np.maximum(den, 1e-30), d).astype(np.float32)
    return t, d


def rule_sm(x, iters=8):
    """scale matching: mean|d*t| == mean|x| (fixed point)."""
    d = np.abs(x).max(axis=1).astype(np.float32)
    t = np.zeros_like(x)
    for _ in range(iters):
        t = np.clip(np.rint(x / d[:, None]), -1, 1)
        mt = np.abs(t).mean(axis=1)
        mx = np.abs(x).mean(axis=1)
        nd = np.where(mt > 0, mx / np.maximum(mt, 1e-30), d)
        d = np.where(nd > 0, nd, d).astype(np.float32)
    return t, d


def rule_robust(x, ratio=1.37, iters=2):
    """Outlier-robust scale: d = ratio * mean|x|, then one least-squares refit.

    The reference's own statistics say its d tracks mean|x| (cv 2.0%) and not
    max|x| (cv 12.7%) -- i.e. the scale is NOT driven by the largest value in the
    group.  That is the signature of value-range protection: outliers are tolerated
    instead of being allowed to set the scale and zero out everything else.
    """
    d = (ratio * np.abs(x).mean(axis=1)).astype(np.float32)
    d = np.maximum(d, np.float32(1e-30))
    for _ in range(iters):
        t = np.clip(np.rint(x / d[:, None]), -1, 1)
        num = (x * t).sum(axis=1)
        den = (t * t).sum(axis=1)
        d = np.where(den > 0, np.abs(num) / np.maximum(den, 1e-30), d).astype(np.float32)
    return np.clip(np.rint(x / d[:, None]), -1, 1), d


r = GGUFReader(GGUF)
names = [t.name for t in r.tensors if P.FOLDABLE.fullmatch(t.name)
         and P.SKIP_FOLD.fullmatch(t.name) is None][:10]

acc = {}
for t in r.tensors:
    if t.name not in set(names):
        continue
    w = P.dequant(t)
    if w.ndim != 2 or w.shape[1] % BLOCK:
        continue
    f = PP.fold_weight(w, P.sign_vector(w.shape[1]), BLOCK).astype(np.float32)
    x = f.reshape(-1, Q.QK)
    for tag, fn in (("ls(MSE)", rule_ls), ("robust", rule_robust)):
        tq, d = fn(x)
        e = float(((d[:, None] * tq - x) ** 2).sum())
        a = acc.setdefault(tag, [0.0, 0, 0.0])
        a[0] += e
        a[1] += x.size
        a[2] += float((tq != 0).mean()) * x.size

print(f"folded rows scored: {acc['ls(MSE)'][1]}")
base = acc["ls(MSE)"][0] / acc["ls(MSE)"][1]
for tag in ("ls(MSE)", "robust"):
    e, n, nz = acc[tag]
    mse = e / n
    print(f"  {tag:9s} MSE={mse:.6e}  ({mse / base:5.3f}x)  nonzero={nz / n * 100:5.2f}%")
print("\nreference Ternary-Bonsai-2-27B sits at 67.2-67.3% non-zero on every tensor.")

# sweep the robust coefficient towards the reference's non-zero fraction
print("\n  ratio   MSE            nonzero")
for ratio in (1.37, 1.6, 1.9, 2.2, 2.6):
    tot_e = tot_n = tot_nz = 0.0
    for t in r.tensors:
        if t.name not in set(names):
            continue
        w = P.dequant(t)
        if w.ndim != 2 or w.shape[1] % BLOCK:
            continue
        f = PP.fold_weight(w, P.sign_vector(w.shape[1]), BLOCK).astype(np.float32)
        x = f.reshape(-1, Q.QK)
        tq, d = rule_robust(x, ratio=ratio)
        tot_e += float(((d[:, None] * tq - x) ** 2).sum())
        tot_n += x.size
        tot_nz += float((tq != 0).mean()) * x.size
    print(f"  {ratio:5.2f}   {tot_e / tot_n:.6e}   {tot_nz / tot_n * 100:5.2f}%")
