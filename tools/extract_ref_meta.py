"""Extract just the scale/sign metadata from the reference GGUF.

The decisive comparison needs (a) the reference's per-128-group d for a few
tensors and (b) its sign vectors.  Both are tiny, so shipping them to the machine
that holds the true weights avoids moving a 5.5 GB file.

Writes small .npy files into remote_extract/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")
from gguf import GGUFReader  # noqa: E402

REF = r"D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
OUT = r"D:\Project\openhanako\workbench\remote_extract"
QK = 128

TENSORS = [
    "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
    "blk.1.ffn_gate.weight", "blk.1.ffn_down.weight",
    "blk.3.attn_q.weight", "blk.3.attn_output.weight",
    "blk.5.ffn_gate.weight", "blk.10.ffn_up.weight", "blk.20.ffn_down.weight",
]

os.makedirs(OUT, exist_ok=True)
r = GGUFReader(REF)

# --- sign vectors ---
widths = [int(v) for v in r.fields["prism.hadamard.sign_widths"].contents()]
vals = np.array([int(v) for v in r.fields["prism.hadamard.sign_values"].contents()],
                dtype=np.float32)
off = 0
signs = {}
for w in widths:
    signs[w] = vals[off:off + w].astype(np.float32)
    np.save(os.path.join(OUT, f"sign_{w}.npy"), signs[w])
    off += w
print(f"sign vectors: {[(w, len(v)) for w, v in signs.items()]}")

meta = {"block": int(r.fields["prism.hadamard.block_size"].contents()),
        "widths": widths, "tensors": {}}

for name in TENSORS:
    t = next((x for x in r.tensors if x.name == name), None)
    if t is None:
        print(f"  {name}: absent")
        continue
    ne = list(t.shape)
    nin, nout = int(ne[0]), int(ne[1])
    ngrp = nin // QK
    b = np.frombuffer(np.ascontiguousarray(t.data).tobytes(), dtype=np.uint8)
    b = b.reshape(nout, ngrp, 28)
    d = b[:, :, 26:28].copy().view(np.float16).reshape(nout, ngrp).astype(np.float32)
    np.save(os.path.join(OUT, "d_" + name.replace(".", "_") + ".npy"), d)
    meta["tensors"][name] = [nin, nout, ngrp]
    print(f"  {name:28s} ne={ne}  d mean={d.mean():.6g}  groups={ngrp}")

import json
with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
sizes = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT))
print(f"\nwrote {OUT}: {len(os.listdir(OUT))} files, {sizes / 1e6:.2f} MB")
