"""Dump the first bytes of one tensor block from each file, to compare encodings."""
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "blk.0.ffn_gate.weight"
a = GGUFReader("/tmp/d27.gguf")
b = GGUFReader("/tmp/bonsai2-ptq10.gguf")
ta = [t for t in a.tensors if t.name == NAME][0]
tb = [t for t in b.tensors if t.name == NAME][0]
print(f"shapes: mine {tuple(ta.shape)}  ref {tuple(tb.shape)}")
print(f"types : mine {ta.tensor_type}  ref {tb.tensor_type}")
ra = np.asarray(ta.data).tobytes()
rb = np.asarray(tb.data).tobytes()
print(f"bytes : mine {len(ra)}  ref {len(rb)}  ratio {len(ra) / max(len(rb), 1):.4f}")
for i in range(3):
    print(f"  mine blk{i} {ra[i * 28:(i + 1) * 28].hex()}")
    print(f"  ref  blk{i} {rb[i * 28:(i + 1) * 28].hex()}")

# byte-multiset overlap on the first 4096 blocks, which survives a permutation
na = np.frombuffer(ra[:4096 * 28], dtype=np.uint8)
nb = np.frombuffer(rb[:4096 * 28], dtype=np.uint8)
print(f"\nfirst 4096 blocks: identical bytes {float((na == nb).mean()) * 100:.2f}%")
