"""Is the reference's zeroing a threshold rule or a rank rule?

A magnitude threshold gives a per-block non-zero count that fluctuates
binomially: n=128, p=0.672 -> sd = sqrt(128*0.672*0.328) = 5.31.

A "keep the top k per block" rule pins the count, so the sd collapses toward 0
(only quantization ties can move it), and a constant 67.2% falls out for free.

This needs only the reference trits, so it runs locally.
"""
import numpy as np

s = np.load(r"D:\Project\openhanako\workbench\ref_trits.npz")
QK = 128
expected_sd = np.sqrt(QK * 0.672 * 0.328)

print(f"binomial expectation for n={QK}, p=0.672: sd = {expected_sd:.2f}\n")
print(f"{'tensor':28s} {'blocks':>7s} {'mean':>7s} {'sd':>7s} {'min':>5s} {'max':>5s}")

allc = []
for key in sorted(s.files):
    if key.endswith("__d"):
        continue
    t = s[key]
    if t.ndim != 2 or t.shape[1] % QK:
        continue
    counts = (t.reshape(t.shape[0], -1, QK) != 0).sum(axis=2).ravel()
    allc.append(counts)
    print(f"{key:28s} {counts.size:7d} {counts.mean():7.2f} {counts.std():7.2f} "
          f"{counts.min():5d} {counts.max():5d}")

c = np.concatenate(allc)
print(f"\n{'ALL':28s} {c.size:7d} {c.mean():7.2f} {c.std():7.2f} {c.min():5d} {c.max():5d}")
print(f"mean fraction = {c.mean() / QK * 100:.2f}%   (reference reports 67.2%)")
print(f"observed sd {c.std():.2f} vs binomial {expected_sd:.2f}  "
      f"-> ratio {c.std() / expected_sd:.3f}")
print("\nratio near 1.0 means a threshold; near 0 means a fixed count per block.")
