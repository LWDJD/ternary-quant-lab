"""Do the F16-stored SSM scalars overflow?

The type diff showed ssm_alpha and ssm_beta stored as F16 in my pack versus BF16 in
the reference.  I dismissed that as "equivalent precision", which is wrong in one
respect that matters: F16 saturates at 65504 while BF16 reaches 3.4e38.  A state
scalar above 65504 stored as F16 becomes infinity, which would corrupt the
recurrent GDN state and produce exactly the observed pattern -- fluent opening
tokens, then drift into repetition.

Reports the actual magnitude range of every SSM-related tensor in the checkpoint.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import torch
from safetensors import safe_open

PATTERNS = ["linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d",
            "linear_attn.norm", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
            "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
            "linear_attn.out_proj", "ssm", "alpha", "beta"]

found = {}
for shard in sorted(glob.glob("/tmp/moe35/*.safetensors")):
    try:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if "layers.0." not in k:
                    continue
                if any(p.split(".")[-1] in k for p in PATTERNS):
                    if k not in found:
                        found[k] = f.get_tensor(k)
    except Exception:  # noqa: BLE001
        continue
    if len(found) > 24:
        break

print(f"{'tensor':58s} {'max|v|':>12s} {'min v':>12s} {'max v':>12s}")
over = []
for k in sorted(found):
    t = found[k].to(torch.float32).numpy().ravel()
    mx = float(np.abs(t).max())
    flag = ""
    if mx > 65504:
        flag = "  <- EXCEEDS F16 MAX"
        over.append(k)
    print(f"{k.split('layers.0.')[-1]:58s} {mx:12.4g} {t.min():12.4g} {t.max():12.4g}{flag}")

print()
if over:
    print(f"TENSORS THAT OVERFLOW F16 ({len(over)}):")
    for k in over:
        print("   ", k)
    print("\nThese must be stored at BF16 (or F32), not F16.")
else:
    print("No SSM tensor exceeds the F16 maximum, so overflow is not the mechanism.")
print("\nFor reference: F16 max 65504, BF16 max 3.39e38.")
