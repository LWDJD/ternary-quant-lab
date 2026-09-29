"""Is my encoder the exact inverse of my decoder?

Take real reference blocks, decode them to trits with my decoder, then re-encode
those exact trits with my encoder (feeding trit*d so the internal round() is
exact) and compare the 28 bytes.  They must be identical if the two are inverses.

No numpy decoding of huge tensors and no packing: a few hundred blocks decide it.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-pack")
import ptq1_0 as Q  # noqa: E402

QK = 128
NAME = sys.argv[1] if len(sys.argv) > 1 else "blk.0.ffn_gate.weight"

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

r = GGUFReader("/tmp/bonsai2-ptq10.gguf")
t = [x for x in r.tensors if x.name == NAME][0]
raw = np.asarray(t.data).tobytes()
nb = len(raw) // 28
print(f"{NAME}: {nb} blocks")

N = 400
same = 0
for i in range(min(N, nb)):
    blk = raw[i * 28:(i + 1) * 28]
    d = np.float32(np.frombuffer(blk[26:28], dtype=np.float16)[0])
    vals = Q.dequantize_row_ptq1_0(blk, QK)
    trits = np.where(np.abs(vals) > 0, np.sign(vals), 0).astype(np.float32)
    # trit*d makes x/d exactly the trit value, so round() is exact and the
    # encoder's d == amax == d, i.e. the same scale the reference stored
    x = (trits * d).astype(np.float32)
    mine = Q.quantize_block_ptq1_0(x)
    if mine[:26] == blk[:26]:
        same += 1
    elif i < 3:
        print(f"  blk{i} ref {blk[:26].hex()}")
        print(f"  blk{i} mine {mine[:26].hex()}")
        print(f"  blk{i} trits {trits[:16].astype(int).tolist()}")

print(f"\ntrit bytes reproduced: {same}/{min(N, nb)}  "
      f"({same / min(N, nb) * 100:.2f}%)")
print("100% means the encoder inverts the decoder exactly.")
print("near 0% means they disagree on the bit layout, which is the bug.")
