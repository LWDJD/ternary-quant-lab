"""Is the alternating least-squares fit actually finding the optimum?

Given the ternary assignment t = round(x/d), the optimal scale is closed form:
d* = <x,t>/<t,t>.  So the whole problem is one-dimensional in d, and a sweep over
d finds the global optimum outright.  If the alternating iteration is landing in a
local minimum, the sweep will show a lower error -- and, importantly, at a
different non-zero fraction.

Reference point: PrismML's own Ternary-Bonsai-2-27B sits at 67.2% non-zero on
every tensor.  The alternating fit gives ~50.8%.  If the sweep's optimum sits near
67%, then the fit is simply suboptimal and that is the whole story.
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-pack")
sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")

import pack_gguf as P  # noqa: E402
import prism_pack as PP  # noqa: E402
import ptq1_0 as Q  # noqa: E402
from gguf import GGUFReader  # noqa: E402

BLOCK = 128
MAX_TENSORS = 10
GGUF = r"D:\AI\lmstudio\models\lmstudio-community\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-Q3_K_L.gguf"


def mse_for(x, d):
    """x: (n,128) ; d: (n,) -> (mse, nonzero fraction) using the closed-form refit."""
    t = np.clip(np.rint(x / d[:, None]), -1.0, 1.0)
    num = (x * t).sum(axis=1)
    den = (t * t).sum(axis=1)
    d2 = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    e = ((d2[:, None] * t - x) ** 2).sum()
    nz = float((t != 0).sum()) / t.size
    return e, nz, d2


r = GGUFReader(GGUF)
names = [t.name for t in r.tensors if P.FOLDABLE.fullmatch(t.name)
         and P.SKIP_FOLD.fullmatch(t.name) is None][:MAX_TENSORS]

rows_all, amax_all = [], []
for t in r.tensors:
    if t.name not in set(names):
        continue
    w = P.dequant(t)
    if w.ndim != 2 or w.shape[1] % Q.QK or w.shape[1] % BLOCK:
        continue
    f = PP.fold_weight(w, P.sign_vector(w.shape[1]), BLOCK).astype(np.float32)
    rr = f.reshape(-1, Q.QK)
    rows_all.append(rr)
    amax_all.append(np.abs(rr).max(axis=1))
X = np.concatenate(rows_all)
AM = np.concatenate(amax_all)
n = X.shape[0]
print(f"folded rows: {n} (from {len(rows_all)} tensors, block {BLOCK})")

t0 = time.time()
# --- alternating fit (current implementation) ---
cur = Q.quantize_blocks_ptq1_0(X, mode="ls")
deq = np.frombuffer(cur, dtype=np.uint8).reshape(n, Q.BLOCK_BYTES)
out = np.stack([Q.dequantize_row_ptq1_0(deq[i].tobytes(), Q.QK) for i in range(n)])
cur_mse = float(((out - X) ** 2).sum())
cur_nz = float((np.abs(out) > 1e-12).sum()) / X.size
print(f"\ncurrent alternating fit : MSE={cur_mse:.6e}  nonzero={cur_nz * 100:5.2f}%")

# --- sweep d = alpha * amax ---
best = None
print("\n  alpha      MSE            nonzero")
for alpha in np.concatenate([np.arange(0.20, 0.62, 0.02), np.arange(0.62, 1.01, 0.05)]):
    d = (AM * alpha).astype(np.float32)
    d = np.maximum(d, np.float32(1e-12))
    e, nz, _ = mse_for(X, d)
    tag = ""
    if best is None or e < best[0]:
        best = (e, alpha, nz)
        tag = "  <-- best so far"
    print(f"  {alpha:5.2f}   {e:.6e}   {nz * 100:5.2f}%{tag}")

print(f"\nbest alpha={best[1]:.2f}  MSE={best[0]:.6e}  nonzero={best[2] * 100:.2f}%")
print(f"vs current      MSE={cur_mse:.6e}  nonzero={cur_nz * 100:.2f}%")
print(f"improvement: {cur_mse / best[0]:.4f}x   ({time.time() - t0:.1f}s)")
