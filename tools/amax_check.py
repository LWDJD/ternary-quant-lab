"""What non-zero fraction does each candidate scale rule actually produce?

Turns the claim "their models cannot have come from the shipped encoder" into a
number.  Same folded weights as theta_sweep.py.
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-pack")
sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")

import pack_gguf as P  # noqa: E402
import prism_pack as PP  # noqa: E402
from gguf import GGUFReader  # noqa: E402

BLOCK = 128
GGUF = (r"D:\AI\lmstudio\models\lmstudio-community"
        r"\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-Q3_K_L.gguf")

r = GGUFReader(GGUF)
rows = []
for t in r.tensors:
    n = t.name
    if not n.endswith(".weight"):
        continue
    w = P.dequant(t)
    if w.ndim != 2 or w.shape[1] % BLOCK:
        continue
    if not any(k in n for k in ("ffn_gate", "ffn_up", "ffn_down",
                                "attn_q", "attn_k", "attn_v", "attn_output")):
        continue
    rows.append(PP.fold_weight(w, P.sign_vector(w.shape[1]), BLOCK)
                .astype(np.float32).reshape(-1, BLOCK))
    if len(rows) >= 6:
        break
x = np.concatenate(rows, axis=0)
mx = np.abs(x).mean(axis=1)
print(f"folded rows {x.shape[0]}  cols {x.shape[1]}")

print("\nrule                                  nonzero%   mean(d/mean|x|)")
for label, d in [
    ("d = amax            (shipped _ref)", np.abs(x).max(axis=1)),
    ("d = 1.37 * mean|x|   (my robust)", np.float32(1.37) * mx),
    ("d = 1.0668 * mean|x| (mean67)", np.float32(1.0668) * mx),
    ("d = least squares", None),
]:
    if d is None:
        dd = np.abs(x).max(axis=1).astype(np.float32)
        t = np.zeros_like(x)
        for _ in range(3):
            t = np.clip(np.rint(x / dd[:, None]), -1, 1)
            num = (x * t).sum(axis=1)
            den = (t * t).sum(axis=1)
            dd = np.where(den > 0, np.abs(num) / np.maximum(den, 1e-30), dd).astype(np.float32)
        d = dd
    d = np.maximum(d.astype(np.float32), 1e-30)
    t = np.clip(np.rint(x / d[:, None]), -1, 1)
    nz = float((t != 0).mean()) * 100
    ratio = float((d / np.maximum(mx, 1e-30)).mean())
    print(f"  {label:36s} {nz:7.2f}    {ratio:6.3f}")

print("\nreference Ternary-Bonsai-2-27B: 67.2% non-zero, and its stored d tracks")
print("mean|x| with cv 2.0% (vs 12.7% for max|x|).")
