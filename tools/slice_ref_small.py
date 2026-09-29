"""Slice a tiny (rows x blocks) corner of ref_trits.npz: trits + per-block d.

4 rows x 5 blocks is 5120 samples, enough to pin a threshold to a few thousandths,
and small enough to cross the shell-size limit.
"""
import numpy as np

src = np.load(r"D:\Project\openhanako\workbench\ref_trits.npz")
ROWS, BLOCKS = 4, 5
COLS = BLOCKS * 128

out = {}
for k in list(src.files):
    if not k.endswith("__d"):
        continue
    base = k[:-3]          # strip "__d"
    if base not in src.files:
        continue
    t = src[base][:ROWS, :COLS]
    d = src[k][:ROWS, :BLOCKS]
    out[base + "__t"] = t
    out[base + "__d"] = d
    print(f"  {base:28s} trits {t.shape}  d {d.shape}  "
          f"nonzero={float((t != 0).mean()) * 100:.2f}%")

np.savez_compressed(r"D:\Project\openhanako\workbench\ref_small.npz", **out)
import os  # noqa: E402
print(f"wrote ref_small.npz  {os.path.getsize(r'D:\Project\openhanako\workbench\ref_small.npz')} B")
