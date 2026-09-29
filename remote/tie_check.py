"""Does the reference keep "all values >= the 86th largest"?

Its per-block non-zero count is min exactly 86, mean 86.05, but max 99.  A plain
rank rule would give exactly 86; a plain threshold would vary both ways.  "Keep
everything at or above the 86th largest" gives 86 plus any ties at the cutoff,
which explains a hard floor of 86 together with an occasional excess.

Ties need a coarse value grid: fp32 continuous values almost never collide, bf16
does.  So measure the tie count at the cutoff under fp32 vs bf16 and compare its
distribution against the reference's excess (count - 86).

Earlier rounds only looked at agreement, never at the count distribution, which is
why this was missed.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/prismwork")
import prism_pack as PP  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader  # noqa: E402

QK, FOLD, K = 128, 1024, 86
POW3 = (1, 3, 9, 27, 81, 243)
REF = "/tmp/ref.gguf"
P = "model.language_model.layers.0."
JOBS = [("blk.0.ffn_gate.weight", P + "mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", P + "mlp.up_proj.weight")]
NROWS = 8


def to_bf16(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).to(torch.float32)
    return t.numpy()


def to_fp16(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).to(torch.float16).to(torch.float32)
    return t.numpy()


def decode_row(raw: bytes, nb: int) -> np.ndarray:
    out = np.empty((nb, QK), dtype=np.int8)
    for i in range(nb):
        blk = raw[i * 28:(i + 1) * 28]
        qs, qh = blk[0:24], blk[24:26]
        v = out[i]
        for c, eoff, boff in ((16, 0, 0), (8, 80, 16)):
            for n in range(5):
                base = eoff + n * c
                for m in range(c):
                    q = (int(qs[boff + m]) * POW3[n]) & 0xFF
                    v[base + m] = (((q * 3) >> 8) & 0xFF) - 1
        for n in range(4):
            for h in range(2):
                q = (int(qh[h]) * POW3[n]) & 0xFF
                v[120 + n * 2 + h] = (((q * 3) >> 8) & 0xFF) - 1
    return out


def load_rows(hf_name, rows):
    for shard in sorted(glob.glob("/tmp/models/qwen38-27b/*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if hf_name in set(f.keys()):
                    return f.get_tensor(hf_name)[:rows].to("cpu", torch.float32).numpy()
        except Exception:  # noqa: BLE001
            continue
    return None


r = GGUFReader(REF)
rmap = {t.name: t for t in r.tensors}
ref_excess, tie32, tie16, tiebf = [], [], [], []

for gguf_name, hf_name in JOBS:
    t = rmap.get(gguf_name)
    if t is None:
        continue
    raw = np.asarray(t.data).tobytes()
    rowbytes = (int(t.shape[0]) // QK) * 28
    nrows = min(NROWS, int(t.shape[1]))
    w = load_rows(hf_name, nrows)
    if w is None:
        continue
    width = w.shape[1]
    sign = np.load(f"/mnt/workspace/prismwork/sign_{width}.npy")
    x32 = PP.fold_weight(w, sign, FOLD).astype(np.float32)
    # rounding must happen AFTER the fold: rounding the input leaves the folded
    # values continuous, which is why the earlier test saw no ties at all
    x32_out = x32.astype(np.float32)
    x16 = to_fp16(x32_out)
    xbf = to_bf16(x32_out)
    x32 = x32_out
    nb = width // QK

    for row in range(nrows):
        ref_t = decode_row(raw[row * rowbytes:(row + 1) * rowbytes], nb)
        for b in range(nb):
            cnt = int((ref_t[b] != 0).sum())
            if cnt:
                ref_excess.append(cnt - K)
            for arr, sink in ((x32, tie32), (x16, tie16), (xbf, tiebf)):
                v = np.abs(arr[row, b * QK:(b + 1) * QK])
                s = np.sort(v)[::-1]
                if s[K - 1] > 0:
                    sink.append(int((v >= s[K - 1]).sum()) - K)
    print(f"  {gguf_name}: done")


def summary(name, a):
    a = np.array(a)
    if not a.size:
        print(f"  {name}: empty")
        return
    print(f"  {name:22s} mean {a.mean():6.3f}  max {a.max():3d}  "
          f"zero {float((a == 0).mean()) * 100:5.1f}%  nonzero {int((a > 0).sum())}")


print(f"\nblocks {len(ref_excess)}")
summary("reference excess", ref_excess)
summary("fp32 ties", tie32)
summary("fp16 ties", tie16)
summary("bf16 ties", tiebf)
print("\nIf a variant's excess distribution lines up with the reference's, the rule")
print("is 'keep everything >= the 86th largest' on that value grid.")
