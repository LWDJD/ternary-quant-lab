"""Does an importance-weighted rank reproduce the reference's kept set?

The reference keeps exactly 86 per 128-block, but its kept set is not a prefix of
my |x| ordering in any of 1280 blocks, and the disagreement even reaches my rank
48 -- which no value perturbation of plausible size can explain.  So the ranking
must use a different quantity.

The natural candidate is an activation-aware score: |w| times the column's
activation energy, i.e. an imatrix.  This ranks by that and re-measures the prefix
rate.  Baseline (|x| alone) was 0/1280.

The imatrix was produced by llama-imatrix on the reference itself, so it is the
activation statistics of the very model under study.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

QK, FOLD, K = 128, 1024, 86
POW3 = (1, 3, 9, 27, 81, 243)
REF = "/tmp/ref.gguf"
IMAT = "/tmp/imat.dat"
P = "model.language_model.layers.0."
JOBS = [("blk.0.ffn_gate.weight", P + "mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", P + "mlp.up_proj.weight")]
NROWS = 8


def decode_row(raw: bytes, nb: int) -> np.ndarray:
    out = np.empty((nb, QK), dtype=np.int8)
    for i in range(nb):
        blk = raw[i * 28:(i + 1) * 28]
        qs, qh = blk[0:24], blk[24:26]
        v = out[i]
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
    return out


def load_rows(hf_name, rows):
    for shard in sorted(glob.glob("/tmp/models/qwen38-27b/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


im = GGUFReader(IMAT)
print(f"imatrix tensors: {len(im.tensors)}")
print("  sample names:", [t.name for t in im.tensors[:4]])
imp = {}
for t in im.tensors:
    key = t.name
    imp[key] = np.asarray(t.data, dtype=np.float32).reshape(-1)
print(f"  importance entries: {len(imp)}")

r = GGUFReader(REF)
rmap = {t.name: t for t in r.tensors}
stats = {"abs": [0, 0], "w_lin": [0, 0], "w_sqrt": [0, 0]}

for gguf_name, hf_name in JOBS:
    t = rmap.get(gguf_name)
    if t is None:
        continue
    raw = np.asarray(t.data).tobytes()
    rowbytes = (int(t.shape[0]) // QK) * 28
    nrows = min(NROWS, int(t.shape[1]))
    w = load_rows(hf_name, nrows)
    if w is None:
        continue
    width = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")
    x = PP.fold_weight(w, sign, FOLD).astype(np.float32)
    nb = width // QK

    col = imp.get(f"{gguf_name}.in_sum2")
    if col is None:
        print(f"  {gguf_name}: no importance entry")
        continue
    if col.size != width:
        print(f"  {gguf_name}: importance {col.size} != width {width}")
        continue
    wlin = col
    wsq = np.sqrt(np.maximum(col, 0))

    for row in range(nrows):
        ref_t = decode_row(raw[row * rowbytes:(row + 1) * rowbytes], nb)
        xr = x[row].reshape(nb, QK)
        for b in range(nb):
            their = ref_t[b] != 0
            k = int(their.sum())
            if k == 0 or k == QK:
                continue
            off = b * QK
            base = np.abs(xr[b])
            for name, score in (("abs", base),
                                ("w_lin", base * wlin[off:off + QK]),
                                ("w_sqrt", base * wsq[off:off + QK])):
                order = np.argsort(-score)
                pr = np.zeros(QK, dtype=bool)
                pr[order[:k]] = True
                stats[name][0] += int((pr == their).all())
                stats[name][1] += 1
    print(f"  {gguf_name}: done")

print(f"\n{'score':10s} {'exact prefix rate':>18s}")
for name, (ok, tot) in stats.items():
    print(f"{name:10s} {ok}/{tot}  ({ok / max(tot, 1) * 100:.2f}%)")
print("\nabs = |x| only (baseline), w_lin = |x|*E[x^2], w_sqrt = |x|*sqrt(E[x^2])")
