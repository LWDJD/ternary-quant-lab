"""Dump N blocks of the 26 trit bytes (no d) as base64, or compare against one.

mode dump : print base64 of the first N blocks' trit bytes
mode cmp  : read a base64 dump and report how many of those blocks match
"""
import base64
import sys

import numpy as np

for p in (r"D:\Project\openhanako\workbench\prism-tip\gguf-py",
          "/mnt/workspace/prismwork/prism-src/gguf-py"):
    sys.path.insert(0, p)
from gguf import GGUFReader  # noqa: E402

mode = sys.argv[1]
path = sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "blk.0.ffn_gate.weight"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 200

r = GGUFReader(path)
t = [x for x in r.tensors if x.name == name][0]
raw = np.asarray(t.data).tobytes()
blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)[:n, :26]

if mode == "dump":
    print(base64.b64encode(blk.tobytes()).decode())
else:
    ref = np.frombuffer(base64.b64decode(sys.argv[5].strip()), dtype=np.uint8).reshape(-1, 26)
    mine = blk[:len(ref)]
    same = int((mine == ref).all(axis=1).sum())
    bytewise = float((mine == ref).mean()) * 100
    print(f"{name}: {same}/{len(ref)} blocks identical ({same / len(ref) * 100:.2f}%)")
    print(f"  bytewise identical: {bytewise:.2f}%")
    # per-element trit agreement, both sides decoded
    print(f"  differing blocks: {len(ref) - same}")
