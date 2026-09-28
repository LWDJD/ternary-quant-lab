"""A/B the two ternary quantizers on real folded weights.

Measures exactly what the quantizer controls: squared reconstruction error in the
*rotation domain* (where PTQ1_0 actually quantizes).  Sign vector, block size and
fold set are held fixed, so any difference is the quantizer alone.

  ls : independent rounding + least-squares scale (current)
  ef : error feedback -- the residual is carried element to element inside the
       block, so errors cancel instead of accumulating.  Post-training, no data.

Only a handful of representative tensors are scored, which keeps it to seconds and
is enough to rank the modes before paying for a full pack.
"""
import re
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-pack")
sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")

import pack_gguf as P  # noqa: E402
import ptq1_0 as Q  # noqa: E402
import prism_pack as PP  # noqa: E402
from gguf import GGUFReader  # noqa: E402

fold_weight = PP.fold_weight

BLOCK = 128
MAX_TENSORS = 10
MODES = [("ls", dict(mode="ls")), ("ef1", dict(mode="ef", iters=1)),
         ("ef3", dict(mode="ef", iters=3))]


def decode(blob: bytes, nb: int) -> np.ndarray:
    deq = np.frombuffer(blob, dtype=np.uint8).reshape(nb, Q.BLOCK_BYTES)
    return np.stack([Q.dequantize_row_ptq1_0(deq[i].tobytes(), Q.QK) for i in range(nb)])


def run(path):
    r = GGUFReader(path)
    names = [t.name for t in r.tensors if P.FOLDABLE.fullmatch(t.name)
             and P.SKIP_FOLD.fullmatch(t.name) is None]
    hit = set(names[:MAX_TENSORS])
    print(f"\n=== {path.split(chr(92))[-1]}")
    print(f"  foldable={len(names)}   scoring={len(hit)}")
    acc = {m: [0.0, 0] for m, _ in MODES}
    nz = {m: [0, 0] for m, _ in MODES}
    t0 = time.time()
    for t in r.tensors:
        if t.name not in hit:
            continue
        w = P.dequant(t)
        if w.ndim != 2 or w.shape[1] % Q.QK or w.shape[1] % BLOCK:
            continue
        f = fold_weight(w, P.sign_vector(w.shape[1]), BLOCK).astype(np.float32)
        rows = f.reshape(-1, Q.QK)
        for m, kw in MODES:
            blob = Q.quantize_blocks_ptq1_0(rows, **kw)
            out = decode(blob, rows.shape[0])
            e = float(((out - rows) ** 2).sum())
            acc[m][0] += e
            acc[m][1] += rows.size
            nz[m][0] += int((np.abs(out) > 1e-12).sum())
            nz[m][1] += rows.size
        print(f"    scored {t.name}")
    print(f"  ({time.time() - t0:.1f}s)")
    base = None
    for m, _ in MODES:
        mse = acc[m][0] / max(1, acc[m][1])
        if base is None:
            base = mse
        print(f"    {m:4s}  MSE={mse:.6e}  ({mse / base:5.3f}x)  nonzero={nz[m][0] / max(1, nz[m][1]) * 100:5.1f}%")


if __name__ == "__main__":
    for p in (sys.argv[1:] or [
        r"D:\AI\lmstudio\models\lmstudio-community\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-Q3_K_L.gguf",
    ]):
        run(p)
