"""Bit-pack the reference sign vectors so they fit through the shell-size limit.

+/-1 per element, so one bit each: 5120 elements -> 640 bytes.
"""
import numpy as np

out = {}
for w in (5120, 6144, 17408):
    v = np.load(rf"D:\Project\openhanako\workbench\remote_extract\sign_{w}.npy")
    bits = np.packbits((v > 0).astype(np.uint8))
    out[f"sign_{w}"] = bits
    print(f"  sign_{w}: n={v.size}  unique={sorted(set(np.unique(v).tolist()))} "
          f"pos={float((v > 0).mean()) * 100:.1f}%  packed={bits.size} B")

np.savez_compressed(r"D:\Project\openhanako\workbench\sign_packed.npz", **out)
print("wrote sign_packed.npz")
