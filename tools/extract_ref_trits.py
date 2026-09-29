"""Extract the reference GGUF's exact trits, so a scale rule can be scored offline.

Rationale
---------
Trit agreement with the reference is a pure function of the quantisation rule.  So
instead of packing a whole model per candidate rule (19 min each), score rules
directly against the reference's own trits.  That needs the reference trits here,
plus the original weights on the box that has them.

Decoding is ported verbatim from quantize_row_ptq1_0_ref / dequantize_row_ptq1_0
(stages {16, 8} over qs[24], then qh[2] -> elements 120..127).  The decoder is
self-validating: the reference is a constant 67.2% non-zero on every tensor, so a
wrong byte layout shows up immediately as the wrong zero fraction.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")
from gguf import GGUFReader  # noqa: E402

REF = (r"D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf"
       r"\Ternary-Bonsai-2-27B-PTQ1_0.gguf")
OUT = r"D:\Project\openhanako\workbench\ref_trits.npz"
ROW_LIMIT = 256          # rows per tensor; rows are independent (blocks run along ne0)
QK = 128
POW3 = (1, 3, 9, 27, 81, 243)

# (tensor, input width) -- all folded by the reference, widths covered by sign vectors
WANT = [
    ("blk.0.ffn_gate.weight", 5120),
    ("blk.0.ffn_up.weight", 5120),
    ("blk.0.attn_q.weight", 5120),
    ("blk.0.attn_output.weight", 6144),
    ("blk.0.ssm_out.weight", 6144),
]


def decode_block(blk: bytes) -> np.ndarray:
    """28-byte PTQ1_0 block -> 128 trits in {-1, 0, +1}."""
    qs, qh = blk[0:24], blk[24:26]
    t = np.zeros(QK, dtype=np.int8)
    # qs: two stages, c=16 over bytes 0..15, c=8 over bytes 16..23
    for c, eoff, boff in ((16, 0, 0), (8, 80, 16)):
        for n in range(5):
            for m in range(c):
                q = (int(qs[boff + m]) * POW3[n]) & 0xFF
                xi = ((q * 3) >> 8) & 0xFF
                t[eoff + n * c + m] = xi - 1
    # qh: 2 bytes -> elements 120..127
    for n in range(4):
        for h in range(2):
            q = (int(qh[h]) * POW3[n]) & 0xFF
            xi = ((q * 3) >> 8) & 0xFF
            t[120 + n * 2 + h] = xi - 1
    return t


r = GGUFReader(REF)
by_name = {t.name: t for t in r.tensors}
print(f"reference tensors: {len(r.tensors)}")

out: dict[str, np.ndarray] = {}
meta: dict[str, dict] = {}

for name, width in WANT:
    t = by_name.get(name)
    if t is None:
        print(f"  missing: {name}")
        continue
    ne0 = int(t.shape[0])
    ne1 = int(t.shape[1])
    if ne0 != width:
        print(f"  width mismatch {name}: expected {width}, got {ne0}")
        continue

    raw = np.asarray(t.data).tobytes()
    rowbytes = (ne0 // QK) * 28
    if len(raw) < rowbytes * ne1:
        print(f"  short read {name}: {len(raw)} < {rowbytes * ne1}")
        continue

    rows = min(ROW_LIMIT, ne1)
    trits = np.empty((rows, ne0), dtype=np.int8)
    scales = np.empty((rows, ne0 // QK), dtype=np.float32)
    for i in range(rows):
        base = i * rowbytes
        row = raw[base:base + rowbytes]
        blocks = [row[b * 28:(b + 1) * 28] for b in range(ne0 // QK)]
        trits[i] = np.concatenate([decode_block(b) for b in blocks])
        # d lives in the last 2 bytes of the block (validated by the round-trip
        # against llama-quantize, which matched every block scale)
        scales[i] = np.frombuffer(b"".join(b[26:28] for b in blocks), dtype=np.float16)

    nz = float((trits != 0).mean()) * 100
    bad_d = int((~np.isfinite(scales) | (scales <= 0)).sum())
    out[name.replace(".", "_")] = trits
    out[name.replace(".", "_") + "__d"] = scales
    meta[name] = {"ne0": ne0, "ne1": ne1, "rows": rows, "nonzero": nz,
                  "bad_d": bad_d, "d_mean": float(scales.mean())}
    flag = "ok" if abs(nz - 67.2) < 3 else "LAYOUT SUSPECT"
    print(f"  {name:32s} ne0={ne0:6d} rows={rows:4d} nonzero={nz:6.2f}%  "
          f"d_mean={scales.mean():.5f} bad_d={bad_d}  {flag}")

np.savez_compressed(OUT, **out)
with open(OUT.replace(".npz", ".json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
print(f"\nwrote {OUT}")
