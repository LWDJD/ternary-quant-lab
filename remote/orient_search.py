"""One-shot parallel search for the exact rotation convention.

Our fold is self-consistent with the fork's runtime (verified bit-exactly on a
synthetic model), but the reference's trits are uncorrelated with our
reconstruction -- 33%, i.e. chance for ternary.  So the reference used a different
convention.  Rather than guess one at a time, enumerate the plausible space and
score every candidate in parallel:

    order      natural (i&j popcount) | bit-reversed rows/cols | gray-coded rows
    scale      1/sqrt(n) | 1/n | 1
    sign_side  before the Hadamard | after it
    blocks     contiguous slices | strided
    sign_idx   global (blk*B+i) | modulo (i)

Scored by trit agreement against the reference's own trits at its own d.
33% = unrelated, ~99% = the right convention.
"""
import itertools
import json
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")

from gguf import GGUFReader  # noqa: E402
from ptq1_0 import dequantize_row_ptq1_0  # noqa: E402

REF = "/tmp/bonsai2-PTQ1_0.gguf"
QK, BLOCK = 128, 1024
GNAME = "blk.0.ffn_gate.weight"
HNAME = "model.language_model.layers.0.mlp.gate_proj.weight"
ROWS = 64


def parity_matrix(n, order):
    idx = np.arange(n, dtype=np.uint32)
    if order == "bitrev":
        b = int(np.ceil(np.log2(n)))
        r = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            v, x = 0, int(i)
            for _ in range(b):
                v = (v << 1) | (x & 1)
                x >>= 1
            r[i] = v
        idx = r
    elif order == "gray":
        idx = (idx ^ (idx >> 1)).astype(np.uint32)
    par = idx[:, None] & idx[None, :]
    p = np.zeros_like(par)
    v = par.copy()
    while v.any():
        p ^= v & 1
        v >>= 1
    return np.where(p & 1, -1.0, 1.0).astype(np.float32)


def hadamard(n, order, scale):
    h = parity_matrix(n, order)
    if scale == "inv_sqrt":
        return (h / np.sqrt(n)).astype(np.float32)
    if scale == "inv_n":
        return (h / n).astype(np.float32)
    return h.astype(np.float32)


def build_variants():
    return [dict(order=o, scale=s, sign_side=ss, blocks=b, sign_idx=si)
            for o in ("natural", "bitrev", "gray")
            for s in ("inv_sqrt", "inv_n", "none")
            for ss in ("before", "after")
            for b in ("contig", "strided")
            for si in ("global", "modulo")]


def score(args):
    v, x, sgn_full, trits, d, nin = args
    nblk = nin // BLOCK
    H = hadamard(BLOCK, v["order"], v["scale"])
    xb = x.reshape(x.shape[0], nblk, BLOCK).astype(np.float32)
    if v["blocks"] == "strided":
        # strided: block g takes columns g, g+nblk, g+2*nblk, ...
        xb = x.reshape(x.shape[0], BLOCK, nblk).transpose(0, 2, 1)
    if v["sign_idx"] == "global":
        S = sgn_full.reshape(nblk, BLOCK) if v["blocks"] == "contig" else \
            sgn_full.reshape(BLOCK, nblk).transpose(1, 0)
    else:
        S = np.tile(sgn_full[:BLOCK], (nblk, 1)) if v["blocks"] == "contig" else \
            np.tile(sgn_full[:nblk], (BLOCK, 1)).transpose(1, 0)
    if v["sign_side"] == "before":
        y = (xb * S[None, :, :]) @ H
    else:
        y = (xb @ H) * S[None, :, :]
    if v["blocks"] == "strided":
        y = y.transpose(0, 2, 1)
    y = y.reshape(x.shape[0], nin // QK, QK)
    tq = np.clip(np.rint(y / d[:, :, None]), -1, 1)
    agree = float((tq == trits).mean())
    nz = float((tq != 0).mean())
    return (agree, nz, v)


def main():
    r = GGUFReader(REF)
    widths = [int(v) for v in r.fields["prism.hadamard.sign_widths"].contents()]
    vals = np.array([int(v) for v in r.fields["prism.hadamard.sign_values"].contents()],
                    dtype=np.float32)
    signs, off = {}, 0
    for w in widths:
        signs[w] = vals[off:off + w]
        off += w

    t = next(x for x in r.tensors if x.name == GNAME)
    nin, nout = int(t.shape[0]), int(t.shape[1])
    ngrp = nin // QK
    blob = np.ascontiguousarray(t.data).tobytes()
    trits = []
    for i in range(min(nout, ROWS)):
        trits.append(np.frombuffer(dequantize_row_ptq1_0(blob[i * ngrp * 28:(i + 1) * ngrp * 28], nin).tobytes(), dtype=np.float32))
    trits = np.stack(trits).reshape(ROWS, ngrp, QK)
    d = np.frombuffer(blob, dtype=np.uint8).reshape(nout, ngrp, 28)[:ROWS, :, 26:28]
    d = d.copy().view(np.float16).reshape(ROWS, ngrp).astype(np.float32)
    tq_ref = np.clip(np.rint(trits / d[:, :, None]), -1, 1)
    print(f"reference: {GNAME} nout={nout} nin={nin}  rows scored={ROWS}  nonzero={(tq_ref != 0).mean() * 100:.2f}%", flush=True)

    import glob
    import torch
    from safetensors import safe_open
    x = None
    for shard in glob.glob("/tmp/q38-27b/*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            if HNAME in list(f.keys()):
                x = f.get_tensor(HNAME).to(torch.float32).numpy()
                break
    assert x is not None, "HF tensor not found"
    if x.shape[-1] != nin:
        x = x.T
    x = np.ascontiguousarray(x[:ROWS])
    print(f"hf tensor: {x.shape}", flush=True)

    variants = build_variants()
    tasks = [(v, x, signs[nin], tq_ref, d, nin) for v in variants]
    print(f"scoring {len(tasks)} conventions on {mp.cpu_count()} cores...", flush=True)
    with mp.Pool(min(48, mp.cpu_count())) as pool:
        out = pool.map(score, tasks)
    out.sort(key=lambda z: -z[0])
    print("\n  agree   nonzero  convention", flush=True)
    for a, nz, v in out[:20]:
        print(f"  {a * 100:6.2f}%  {nz * 100:6.2f}%  {v}", flush=True)
    print(f"\nchance level is 33.3%; best = {out[0][0] * 100:.2f}%", flush=True)
    with open("/tmp/orient_search.json", "w") as f:
        json.dump([(a, nz, v) for a, nz, v in out], f, indent=1)


if __name__ == "__main__":
    main()
