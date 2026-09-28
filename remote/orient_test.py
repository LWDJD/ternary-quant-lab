"""Which fold orientation did the reference actually use?

The ratio test (d/mean, cv 2%) says the scale is mean-normalised, but the induced
non-zero fraction (58.3%) disagrees with the reference's own trits (67.2%), so the
reconstructed rotation-domain values are not exactly what the reference quantised.
A sharper test than any ratio: rebuild the group with each candidate orientation
and compare the *trits* against the reference's own, at its own d.

    A: W' = W . (S H)      (what our packer does)
    B: W' = W . (H S)
    C: W' = (S H) . W      (sign/Hadamard on the output side)
    D: W' = (H S) . W

The orientation whose rounding reproduces the stored trits almost exactly is the
right one; the others land near 50% agreement.
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")

from gguf import GGUFReader  # noqa: E402

REF = "/tmp/bonsai2-PTQ1_0.gguf"
QK, BLOCK = 128, 1024
TENSORS = [
    ("blk.0.ffn_gate.weight", "model.language_model.layers.0.mlp.gate_proj.weight"),
]


def hadamard(n):
    idx = np.arange(n, dtype=np.uint32)
    par = idx[:, None] & idx[None, :]
    p = np.zeros_like(par)
    v = par.copy()
    while v.any():
        p ^= v & 1
        v >>= 1
    return np.where(p & 1, -1.0, 1.0).astype(np.float32) / np.sqrt(n)


def ref_signs(r):
    widths = [int(v) for v in r.fields["prism.hadamard.sign_widths"].contents()]
    vals = np.array([int(v) for v in r.fields["prism.hadamard.sign_values"].contents()],
                    dtype=np.float32)
    out, off = {}, 0
    for w in widths:
        out[w] = vals[off:off + w]
        off += w
    return out


def ref_trits_and_d(r, name):
    """Per 128-group trits t in {-1,0,1} and scale d, from the packed blocks."""
    t = next(x for x in r.tensors if x.name == name)
    nin, nout = int(t.shape[0]), int(t.shape[1])
    ngrp = nin // QK
    raw = np.ascontiguousarray(t.data).tobytes()
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nout, ngrp, 28)
    qs = b[:, :, 0:24]
    qh = b[:, :, 24:26]
    d = b[:, :, 26:28].copy().view(np.float16).reshape(nout, ngrp).astype(np.float32)

    # unpack the base-3 packing: 16 bytes cover elements 0..79 (5 per byte),
    # 8 bytes cover 80..119, qh covers 120..127 (4 per byte).
    def unpack(bytes_, c, count):
        out = np.empty(bytes_.shape[:-1] + (count,), dtype=np.int64)
        x = bytes_.astype(np.uint32) * 256 + 242
        x = x // 243
        # x now holds 5 (or 4) trits packed little-endian within each byte
        digs = np.stack([x // 81, (x // 27) % 3, (x // 9) % 3, (x // 3) % 3, x % 3], axis=-1)
        out[...] = digs.reshape(bytes_.shape[:-1] + (count,)) if c == 5 else digs[..., 1:].reshape(bytes_.shape[:-1] + (count,))
        return out

    t16 = unpack(qs[:, :, 0:16], 5, 80)          # 16 bytes -> 80 trits
    t8 = unpack(qs[:, :, 16:24], 5, 40)          # 8 bytes  -> 40 trits
    th = unpack(qh, 4, 8)                        # 2 bytes  -> 8 trits
    trits = np.concatenate([t16, t8, th], axis=2).astype(np.float32) - 1.0
    assert trits.shape == (nout, ngrp, QK), trits.shape
    return trits, d, nin, nout


def load_hf(name):
    import glob
    import torch
    from safetensors import safe_open
    for shard in glob.glob("/tmp/q38-27b/*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            if name in list(f.keys()):
                return f.get_tensor(name).to(torch.float32).numpy()
    return None


r = GGUFReader(REF)
signs = ref_signs(r)
H = hadamard(BLOCK)
mask = (np.arange(BLOCK) % BLOCK)  # all

for gname, hname in TENSORS:
    trits, d, nin, nout = ref_trits_and_d(r, gname)
    nout = min(nout, 512)                       # keep the whole thing to seconds
    trits, d = trits[:nout], d[:nout]
    w = load_hf(hname)
    if w is None:
        print(f"{gname}: HF missing")
        continue
    x = np.asarray(w, dtype=np.float32)
    if x.shape[-1] != nin:
        x = x.T
    x = x[:nout]
    assert x.shape == (nout, nin), (gname, x.shape, nout, nin)
    sgn = signs[nin]
    nblk = nin // BLOCK

    xb = x.reshape(nout, nblk, BLOCK)
    S = sgn.reshape(nblk, BLOCK)          # sign is width-long, rotation is per block
    variants = {
        "A W.(S H)": (xb * S[None, :, :]) @ H,
        "B W.(H S)": (xb @ H) * S[None, :, :],
    }
    print(f"\n--- {gname}  {x.shape}  nonzero(ref)={((trits != 0).mean() * 100):.2f}%")
    for tag, y in variants.items():
        y = y.reshape(nout, nin // QK, QK)
        tq = np.clip(np.rint(y / d[:, :, None]), -1, 1)
        agree = float((tq == trits).mean())
        nz = float((tq != 0).mean())
        print(f"    {tag}: trit agreement={agree * 100:6.2f}%   nonzero={nz * 100:5.2f}%")
        # also the ratio for this orientation, restricted to correct trits
        m = np.abs(y)
        ratio = (d / np.maximum(m.mean(axis=2), 1e-30))
        print(f"         d/mean = {ratio.mean():.5f}  cv={ratio.std() / abs(ratio.mean()):.4f}")
