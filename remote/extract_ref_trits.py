"""Extract the reference GGUF's exact trits + per-block d (Linux/remote variant).

Trit agreement with the reference is a pure function of the quantisation rule, so
this lets rules be scored offline instead of packing a whole model per candidate.

Decoding is ported verbatim from dequantize_row_ptq1_0: stages {16, 8} over qs[24]
(elements 0..119), then qh[2] (elements 120..127).  Self-validating: the reference
is a constant 67.2% non-zero on every tensor, so a wrong layout is visible at once.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

REF = "/tmp/bonsai2-ptq10.gguf"
OUT = "/mnt/workspace/prismwork/ref_trits.npz"
ROW_LIMIT = 256
QK = 128
POW3 = (1, 3, 9, 27, 81, 243)

WANT = [("blk.0.ffn_gate.weight", 5120), ("blk.0.ffn_up.weight", 5120),
        ("blk.0.ffn_down.weight", 17408), ("blk.0.ssm_out.weight", 6144)]


def decode_block(blk: bytes) -> np.ndarray:
    qs, qh = blk[0:24], blk[24:26]
    t = np.zeros(QK, dtype=np.int8)
    for c, eoff, boff in ((16, 0, 0), (8, 80, 16)):
        for n in range(5):
            for m in range(c):
                q = (int(qs[boff + m]) * POW3[n]) & 0xFF
                t[eoff + n * c + m] = (((q * 3) >> 8) & 0xFF) - 1
    for n in range(4):
        for h in range(2):
            q = (int(qh[h]) * POW3[n]) & 0xFF
            t[120 + n * 2 + h] = (((q * 3) >> 8) & 0xFF) - 1
    return t


r = GGUFReader(REF)
by_name = {t.name: t for t in r.tensors}
print(f"reference tensors: {len(r.tensors)}")

out, meta = {}, {}
for name, width in WANT:
    t = by_name.get(name)
    if t is None:
        print(f"  missing: {name}")
        continue
    ne0, ne1 = int(t.shape[0]), int(t.shape[1])
    if ne0 != width:
        print(f"  width mismatch {name}: {ne0} != {width}")
        continue
    raw = np.asarray(t.data).tobytes()
    rowbytes = (ne0 // QK) * 28
    rows = min(ROW_LIMIT, ne1)
    trits = np.empty((rows, ne0), dtype=np.int8)
    scales = np.empty((rows, ne0 // QK), dtype=np.float32)
    for i in range(rows):
        row = raw[i * rowbytes:(i + 1) * rowbytes]
        blocks = [row[b * 28:(b + 1) * 28] for b in range(ne0 // QK)]
        trits[i] = np.concatenate([decode_block(b) for b in blocks])
        scales[i] = np.frombuffer(b"".join(b[26:28] for b in blocks), dtype=np.float16)
    nz = float((trits != 0).mean()) * 100
    bad = int((~np.isfinite(scales) | (scales <= 0)).sum())
    key = name.replace(".", "_")
    out[key], out[key + "__d"] = trits, scales
    meta[name] = {"ne0": ne0, "ne1": ne1, "rows": rows, "nonzero": nz,
                  "bad_d": bad, "d_mean": float(scales.mean())}
    flag = "ok" if abs(nz - 67.2) < 3 else "LAYOUT SUSPECT"
    print(f"  {name:32s} ne0={ne0:6d} rows={rows:4d} nonzero={nz:6.2f}% "
          f"d_mean={scales.mean():.5f} bad_d={bad}  {flag}")

np.savez_compressed(OUT, **out)
with open(OUT.replace(".npz", ".json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
print(f"wrote {OUT}")

# The sign vectors travel inside the GGUF metadata, so extracting them here means
# nothing large has to cross the wire.
print("\n=== sign vectors from metadata ===")
try:
    fw = r.fields["prism.hadamard.sign_widths"].contents()
    fv = r.fields["prism.hadamard.sign_values"].contents()
    widths = [int(v) for v in np.asarray(fw).reshape(-1)]
    values = np.asarray(fv).reshape(-1).astype(np.float32)
    print(f"  widths={widths}  total_values={values.size}")
    off = 0
    for w in sorted(widths):
        vec = values[off:off + w]
        off += w
        np.save(f"/mnt/workspace/prismwork/sign_{w}.npy", vec)
        uniq = sorted(set(np.unique(vec).tolist()))[:5]
        print(f"  sign_{w}.npy  n={vec.size}  unique={uniq}  "
              f"pos={float((vec > 0).mean()) * 100:.1f}%")
    print(f"  consumed {off} of {values.size}")
except Exception as e:  # noqa: BLE001
    print(f"  failed: {e}")
    for k in r.fields:
        if "hadamard" in k:
            print(f"    field present: {k}")
