"""Read a FULL-ATTENTION layer's shapes from the real checkpoint.

Earlier dumps only showed layers 0-1, which are GDN.  With
full_attention_interval=4 the full-attention layers are 3, 7, 11, ... so layer 3
is the one to copy.  It apparently fuses Q with the output gate (expected width
256 = 2 x 4 heads x 32), which the synthetic model was missing.
"""
import json
import os
import re
import struct

SRC = "/mnt/workspace/prismwork/models/qwen38-27b"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = f.read(n)
    meta = json.loads(head)
    meta.pop("__metadata__", None)
    return meta


shapes = {}
for fn in sorted(os.listdir(SRC)):
    if fn.endswith(".safetensors"):
        shapes.update(read_header(os.path.join(SRC, fn)))

print("=== layer 3 (full attention) ===")
for k, v in sorted(shapes.items()):
    if re.match(r"model\.language_model\.layers\.3\.", k):
        print(f"  {k:66s} {v['shape']}")

print("\n=== layer 2 (should be GDN) ===")
for k, v in sorted(shapes.items()):
    if re.match(r"model\.language_model\.layers\.2\.", k) and "mlp" not in k:
        print(f"  {k:66s} {v['shape']}")
