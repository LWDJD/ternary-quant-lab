"""Decision-threshold sweep on real folded weights.

Motivation
----------
The shipped quantizer uses d = amax and plain round-to-nearest.  On the reference
GGUF the non-zero fraction is a constant 67.2%, but feeding the reference's *own*
d into round-to-nearest reproduces only 58.0% of its trits.  Same d, same weights,
9 points more non-zero: their decision boundary sits below 0.5*d.

So the single number that matters is the effective threshold

    trit = 0 if |x| < tau else sign(x)

expressed in units of the group's own scale.  This script sweeps tau in units of
mean|x| and reports the non-zero fraction and the reconstruction MSE, to see
whether some plausible tau lands on 67.2% -- and what it costs.
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-pack")
sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")

import pack_gguf as P  # noqa: E402
import prism_pack as PP  # noqa: E402
from gguf import GGUFReader  # noqa: E402

BLOCK = 128
GGUF = r"D:\AI\lmstudio\models\lmstudio-community\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-Q3_K_L.gguf"


def load_folded(limit_tensors=6):
    """Load real weights and fold them the same way the packer does."""
    r = GGUFReader(GGUF)
    rows = []
    for t in r.tensors:
        n = t.name
        if not (n.endswith(".weight") and "." in n):
            continue
        w = P.dequant(t)
        if w.ndim != 2 or w.shape[1] % BLOCK or w.shape[1] < BLOCK:
            continue
        # only fold kinds the packer folds
        if not any(k in n for k in ("ffn_gate", "ffn_up", "ffn_down",
                                    "attn_q", "attn_k", "attn_v", "attn_output")):
            continue
        f = PP.fold_weight(w, P.sign_vector(w.shape[1]), BLOCK).astype(np.float32)
        rows.append(f.reshape(-1, BLOCK))
        if len(rows) >= limit_tensors:
            break
    return np.concatenate(rows, axis=0)


def score(x, tau_mult):
    """trit = 0 if |x| < tau*mean|x| (per row), else sign(x); d refit per row."""
    mx = np.abs(x).mean(axis=1, keepdims=True)
    tau = tau_mult * mx
    t = np.where(np.abs(x) < tau, 0.0, np.sign(x)).astype(np.float32)
    num = (x * t).sum(axis=1)
    den = (t * t).sum(axis=1)
    d = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0).astype(np.float32)
    err = ((d[:, None] * t - x) ** 2).mean()
    return float((t != 0).mean()) * 100, err


print("=== loading + folding (SmolLM2 bench) ===")
x = load_folded()
print(f"rows {x.shape[0]}  cols {x.shape[1]}")

print("\ntau/mean|x|   nonzero%     MSE")
ref_mse = None
for mult in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00):
    nz, mse = score(x, mult)
    if mult == 0.50:
        ref_mse = mse
    print(f"  {mult:5.2f}       {nz:6.2f}    {mse:.6e}")

print("\nreference sits at 67.2% non-zero.")

# what mean|x| multiple reproduces it, and what does it cost?
lo, hi = 0.30, 1.20
for _ in range(20):
    mid = (lo + hi) / 2
    nz, _ = score(x, mid)
    if nz > 67.2:
        lo = mid
    else:
        hi = mid
nz, mse = score(x, (lo + hi) / 2)
print(f"\ntau at 67.2%: {((lo + hi) / 2):.4f} * mean|x|   nonzero {nz:.2f}%   MSE {mse:.6e}")
print(f"MSE vs round-at-0.5 (=1.000x baseline {ref_mse:.6e}): {mse / ref_mse:.3f}x")
