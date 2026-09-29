"""Slice a tiny sample out of ref_trits.npz so it can cross the shell-size limit.

Only a few hundred per-block scales are needed to answer "what is d relative to
mean|x|", and the original weights live on the remote box.
"""
import numpy as np

src = np.load(r"D:\Project\openhanako\workbench\ref_trits.npz")
ROWS = 8
out = {}
for k in src.files:
    if not k.endswith("__d"):
        continue
    out[k] = src[k][:ROWS]
    print(f"  {k:28s} {src[k][:ROWS].shape}")

np.savez_compressed(r"D:\Project\openhanako\workbench\ref_d_sample.npz", **out)
print("wrote ref_d_sample.npz")
