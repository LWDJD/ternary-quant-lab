"""Rotator test on the qwen35 graph path.

  plain   : convert_hf_to_gguf.py                 (no transform)
  rotated : prism_pack_hf.py --no-quant           (folded, F16)

The fold is mathematically lossless, so the runtime must produce identical
logits -> identical perplexity.  A difference means the runtime is not applying
the transform across the same tensors/axes the packer assumed, i.e. the rotator
is wrong for this architecture.  This is the one test that separates the rotator
from the quantizer, and it had never been run on a qwen35 model.
"""
import os
import subprocess
import sys

PY = sys.executable
SRC = "/mnt/workspace/prismwork/prism-src"
BIN = "/tmp/llb/bin"
Q35 = "/tmp/q35"
PLAIN = "/tmp/q35-plain.gguf"
ROT = "/tmp/q35-rot.gguf"
CORPUS = "/mnt/workspace/prismwork/prism-src/README.md"


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


print("=== 1. plain F16 via the fork's own converter ===")
rc, out = sh(f"cd {SRC} && PYTHONPATH={SRC}/gguf-py {PY} -u convert_hf_to_gguf.py "
             f"{Q35} --outtype f16 --outfile {PLAIN} 2>&1 | tail -3")
print(f"  rc={rc}  {out.strip()[-300:]}")

print("\n=== 2. rotated F16 via our packer ===")
rc, out = sh(f"cd /mnt/workspace/prismwork && {PY} -u prism_pack_hf.py {Q35} {ROT} "
             f"--block 128 --no-quant --fork {SRC} 2>&1 | tail -5")
print(f"  rc={rc}  {out.strip()[-500:]}")

print("\n=== 3. perplexity on both (must match exactly) ===")
res = {}
for tag, path in (("plain", PLAIN), ("rotated", ROT)):
    if not os.path.exists(path):
        print(f"  {tag}: MISSING")
        continue
    rc, out = sh(f"{BIN}/llama-perplexity -m {path} -c 64 -f {CORPUS} 2>&1 | tail -4")
    line = [l for l in out.splitlines() if "Final estimate" in l]
    res[tag] = line[0].strip() if line else "??"
    print(f"  {tag:8s} rc={rc}  {res.get(tag)}")

if len(res) == 2:
    same = res["plain"].split("PPL =")[-1].strip() == res["rotated"].split("PPL =")[-1].strip()
    print(f"\n  >>> IDENTICAL: {same}")
    print("      True  -> rotator OK for qwen35, the defect is in the quantizer")
    print("      False -> the rotator itself is wrong for this architecture")
