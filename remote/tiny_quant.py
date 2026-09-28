"""Three-way on the tiny qwen35: F16 / rotated F16 / PTQ1_0.

This is the fast iteration loop that was missing: a GDN-bearing qwen35 that
builds and scores in seconds, so recipe variants can be A/B'd cheaply instead of
paying 20 minutes plus a 10 GB download each time.

Baseline so far:
    plain F16    PPL 300799.99
    rotated F16  PPL 300801.53   (rotator correct: 5e-6 is F16 rounding)
"""
import json
import os
import subprocess
import sys

BIN = "/tmp/llb/bin"
SRC = "/mnt/workspace/prismwork/prism-src"
Q35 = "/tmp/q35"
CORPUS = "/mnt/workspace/prismwork/prism-src/README.md"


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


def ppl(path):
    rc, out = sh(f"{BIN}/llama-perplexity -m {path} -c 128 -f {CORPUS} --chunks 3 2>&1 | tail -3")
    line = [l for l in out.splitlines() if "Final estimate" in l]
    if line:
        return line[0].split("PPL =")[-1].strip().split("+/-")[0].strip()
    err = [l for l in out.splitlines() if " E " in l]
    return "ERR " + (err[0][:120] if err else out.strip()[-120:])


VARIANTS = [
    ("ptq10-default", ["--block", "128"]),
    ("ptq10-density672", ["--block", "128", "--density", "0.672"]),
]

out_rows = []
for tag, extra in VARIANTS:
    dst = f"/tmp/q35-{tag}.gguf"
    cmd = (f"cd /mnt/workspace/prismwork && {sys.executable} -u prism_pack_hf.py {Q35} {dst} "
           f"--fork {SRC} " + " ".join(extra) + " 2>&1 | tail -2")
    rc, out = sh(cmd)
    ok = os.path.exists(dst)
    val = ppl(dst) if ok else "BUILD FAILED"
    print(f"### {tag}  rc={rc}  PPL={val}")
    print(f"    {out.strip()[-200:]}")
    out_rows.append((tag, val))

print("\n=== summary ===")
print("  plain F16     300799.99")
print("  rotated F16   300801.53   <- rotator correct")
for tag, val in out_rows:
    print(f"  {tag:22s} {val}")
