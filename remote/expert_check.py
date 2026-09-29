"""Does the rotation keep experts separate?

The MoE expert tensors are 3D.  Reading fold_weight says it only touches the last
axis (the input dim), so each expert should get its own blockwise Hadamard and the
quantiser's reshape(-1, 128) should stay inside an expert.  But that depends on an
axis-order assumption I have never tested, and if it fails then every expert is
corrupted at once, which would match the magnitude of the damage.

Test: fold a 3D expert tensor normally, then fold it again after negating the sign
vector.  If experts are independent, the negation changes every expert.  Then fold
with a sign vector that is flipped for ONE expert only -- impossible with a single
vector, so instead check the structural property directly: verify that changing
one expert's input slice changes only that expert's output slice.

Also verify the invariant that makes the runtime scheme work: the folded tensor
must satisfy fold(W)[e] == (W[e] * signs) @ H for each expert e.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402

QK, FOLD = 128, 1024
NAME = "model.language_model.layers.0.mlp.gate_proj.weight"


def load_any(pattern, rows=4):
    for shard in sorted(glob.glob("/tmp/moe35/*.safetensors")):
        try:
            from safetensors import safe_open
            with safe_open(shard, framework="pt") as f:
                for k in f.keys():
                    if pattern in k:
                        return k, f.get_tensor(k).to("cpu", __import__("torch").float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None, None


# the MoE checkpoint stores experts stacked; find the 3D gate tensor
key = None
w = None
for shard in sorted(glob.glob("/tmp/moe35/*.safetensors")):
    try:
        from safetensors import safe_open
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.endswith("mlp.experts.gate_proj.weight") or "experts" in k and "gate" in k:
                    t = f.get_tensor(k)
                    if t.ndim == 3:
                        key, w = k, t[:2].to("cpu", __import__("torch").float32).numpy()
                        break
    except Exception:  # noqa: BLE001
        continue
    if key:
        break

if key is None:
    print("no 3D expert tensor found; dumping candidate names")
    from safetensors import safe_open
    for shard in sorted(glob.glob("/tmp/moe35/*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if "experts" in k:
                    print("  ", k, tuple(f.get_slice(k).get_shape()))
            break
    sys.exit(0)

print(f"tensor {key}  shape {w.shape}  (expert, out, in) assumed")
n_exp, n_out, n_in = w.shape
print(f"experts {n_exp}  out {n_out}  in {n_in}  in%block={n_in % FOLD}")

sign = np.ones(n_in, np.float32)
x = PP.fold_weight(w, sign, FOLD).astype(np.float32)
print(f"folded shape {x.shape}")

# 1. per-expert reconstruction identity
H = np.array([[(-1) ** (bin(i & j).count("1")) for j in range(FOLD)] for i in range(FOLD)],
             dtype=np.float32) / np.sqrt(FOLD)
ok = 0
for e in range(min(2, n_exp)):
    ref = (w[e] * sign[None, :]).reshape(n_out, n_in // FOLD, FOLD) @ H
    ref = ref.reshape(n_out, n_in)
    if np.allclose(ref, x[e], atol=1e-4):
        ok += 1
print(f"per-expert identity holds for {ok}/{min(2, n_exp)} experts")

# 2. does perturbing one expert's input leave other experts untouched?
w2 = w.copy()
w2[0] *= -1.0                      # flip expert 0 only
x2 = PP.fold_weight(w2, sign, FOLD).astype(np.float32)
d0 = float(np.abs(x2[0] - x[0]).max())
d1 = float(np.abs(x2[1] - x[1]).max())
print(f"\nflipping expert 0's input:")
print(f"  change in expert 0: {d0:.6f}")
print(f"  change in expert 1: {d1:.6e}   <- must be 0 if experts are separate")

# 3. does changing the sign vector affect the block layout across experts?
s2 = sign.copy()
s2[:FOLD] *= -1.0
x3 = PP.fold_weight(w, s2, FOLD).astype(np.float32)
print(f"\nflipping signs of the first block only:")
print(f"  change in block 0 of expert 0: {float(np.abs(x3[0, :, :FOLD] - x[0, :, :FOLD]).max()):.6f}")
print(f"  change in block 1 of expert 0: {float(np.abs(x3[0, :, FOLD:] - x[0, :, FOLD:]).max()):.6e}")
