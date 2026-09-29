"""Score quantisation rules against the reference, by trit agreement.

Trit agreement is a pure function of the rule, so this replaces "pack the whole
model and look at PPL" (19 min per candidate) with "fold a few tensors and count
matches" (seconds).

Two questions, deliberately separated:

  A. Given the reference's OWN per-block d, what decision threshold reproduces its
     trits?  This isolates the decision rule from the scale rule.
  B. How well do candidate scale rules do, each with a plain 0.5*d boundary?

Inputs: ref_trits.npz (trits + reference d), the sign vectors, and the original
bf16 weights straight from the HF checkpoint.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
from safetensors import safe_open  # noqa: E402

QK = 128
HF = "/tmp/q38-27b"
NPZ = "/mnt/workspace/prismwork/ref_trits.npz"
SIGNS = {"sign_5120.npy": 5120, "sign_6144.npy": 6144}

# gguf name -> HF name (only folded kinds, only what the reference stores as PTQ1_0)
HFNAMES = {
    "blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj.weight",
    "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
    "blk.0.ssm_out.weight": "model.layers.0.linear_attn.out_proj.weight",
}


def load_signs() -> dict[int, np.ndarray]:
    out = {}
    for fn, width in SIGNS.items():
        try:
            out[width] = np.load(f"/mnt/workspace/prismwork/{fn}").astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  sign {fn}: {e}")
    return out


def load_hf_rows(hf_name: str, rows: int) -> np.ndarray | None:
    """Read the first `rows` rows of a tensor straight out of the shards, as f32."""
    import glob
    for shard in sorted(glob.glob(f"{HF}/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                keys = list(f.keys())
                if hf_name in keys:
                    t = f.get_tensor(hf_name)
                    return t[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


def trits_from(x: np.ndarray, d: np.ndarray, theta: float) -> np.ndarray:
    """Row-wise ternary decision: 0 when |x| < theta*d."""
    out = np.zeros_like(x, dtype=np.int8)
    th = (theta * d)[:, None]
    rel = np.abs(x) >= th
    out[rel] = np.sign(x[rel]).astype(np.int8)
    return out


def scale_amax(x):
    return np.abs(x).max(axis=1)


def scale_ls(x, iters=3):
    d = np.abs(x).max(axis=1).astype(np.float32)
    t = np.zeros_like(x)
    for _ in range(iters):
        t = np.clip(np.rint(x / d[:, None]), -1, 1)
        num = (x * t).sum(axis=1)
        den = (t * t).sum(axis=1)
        d = np.where(den > 0, np.abs(num) / np.maximum(den, 1e-30), d).astype(np.float32)
    return d


ref = np.load(NPZ)
signs = load_signs()
print(f"sign vectors loaded: {sorted(signs)}")

import torch  # noqa: E402  (after the sign load, so a missing torch fails visibly)

report = {}
for gguf_name, hf_name in HFNAMES.items():
    key = gguf_name.replace(".", "_")
    if key not in ref:
        print(f"  no reference trits for {gguf_name}")
        continue
    tgt = ref[key]
    dref = ref[key + "__d"].astype(np.float32)
    rows, ne0 = tgt.shape
    if ne0 not in signs:
        print(f"  no sign vector for width {ne0}")
        continue

    w = load_hf_rows(hf_name, rows)
    if w is None:
        print(f"  HF tensor not found: {hf_name}")
        continue
    if w.shape[1] != ne0:
        print(f"  width mismatch on {hf_name}: HF {w.shape[1]} vs ref {ne0}")
        continue

    fd = PP.fold_weight(w, signs[ne0], QK).astype(np.float32)
    print(f"\n=== {gguf_name}  rows={rows} ne0={ne0} ===")

    # A. their own d, sweeping the decision threshold
    print("  A. reference scale, threshold sweep")
    best = (-1.0, None)
    for theta in (0.25, 0.30, 0.35, 0.39, 0.40, 0.45, 0.50):
        agree = float((trits_from(fd, dref, theta) == tgt).mean()) * 100
        mark = ""
        if agree > best[0]:
            best = (agree, theta)
            mark = "  <- best"
        print(f"       theta={theta:4.2f}  agreement={agree:6.2f}%{mark}")

    # B. candidate scale rules, plain 0.5*d boundary
    print("  B. candidate scale rules (theta=0.5)")
    cands = {
        "amax (shipped)": lambda x: scale_amax(x),
        "ls (mine)": lambda x: scale_ls(x),
    }
    for a in (0.9, 1.0, 1.0668, 1.15, 1.37, 1.5):
        cands[f"d={a}*mean|x|"] = (lambda a_: (lambda x: a_ * np.abs(x).mean(axis=1)))(a)
    for label, fn in cands.items():
        d = fn(fd).astype(np.float32)
        nz = float((trits_from(fd, d, 0.5) != 0).mean()) * 100
        agree = float((trits_from(fd, d, 0.5) == tgt).mean()) * 100
        print(f"       {label:18s} nonzero={nz:6.2f}%  agreement={agree:6.2f}%")

    report[gguf_name] = {"best_theta": best[1], "best_agreement": best[0],
                         "ref_nonzero": float((tgt != 0).mean()) * 100}

print("\n=== summary ===")
print(json.dumps(report, indent=1))
print("\nAgreement chance level is 33.3%. The reference's non-zero fraction is 67.2%.")
