"""Print the first N blocks of one tensor as hex, for cross-machine comparison.

Usage: python hex_sample.py <gguf> <tensor-name> [nblocks]
Works with either GGUF, so the same script runs locally on the reference and
remotely on my pack, and the two hex strings can be diffed directly.
"""
import sys

import numpy as np

GGUF = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "blk.0.ffn_gate.weight"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24

for p in (r"D:\Project\openhanako\workbench\prism-tip\gguf-py",
          "/mnt/workspace/prismwork/prism-src/gguf-py"):
    sys.path.insert(0, p)
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(GGUF)
t = [x for x in r.tensors if x.name == NAME]
if not t:
    print(f"NO TENSOR {NAME}")
    print("first names:", [x.name for x in r.tensors[:5]])
    sys.exit(1)
t = t[0]
raw = np.asarray(t.data).tobytes()
print(f"# {NAME} type={t.tensor_type} shape={tuple(t.shape)} bytes={len(raw)}")
for i in range(min(N, len(raw) // 28)):
    print(raw[i * 28:(i + 1) * 28].hex())
