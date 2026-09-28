"""Short remote test of the block-512 pack.

Remote CPU runs at ~2.2 tok/s, so keep it to two facts and a short generation.
Baselines already measured:
    healthy IQ2_M           facts 4/4   uniq 0.72  rep6  1
    qwen36-moe-b1024        facts 4/4   uniq 0.24  rep6 16   (fallback 80: ffn_down_exps -> Q4_0)
    moe35-512 (this one)    fallback 0 -- nothing left unrotated
"""
import os
import re
import subprocess
import time

BIN = "/tmp/llb/bin/llama-completion"
M = "/tmp/moe35-512.gguf"

FACTS = [
    ("france", "The capital of France is", "Paris"),
    ("water", "Water is made of hydrogen and", "oxygen"),
]


def run(prompt, ntok):
    args = [BIN, "-m", M, "-ngl", "0", "-c", "4096", "-p", prompt, "-n", str(ntok),
            "--temp", "0", "--seed", "1", "--jinja", "-st", "--no-display-prompt",
            "--threads", "64"]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True, timeout=3000,
                       errors="replace")
    return (r.stdout or ""), time.time() - t0


print(f"### file: {os.path.getsize(M)/2**30:.2f} GiB")
print("### facts (n=220)")
hits = 0
for pid, p, want in FACTS:
    body, dt = run(p, 220)
    ok = want.lower() in body.lower()
    hits += ok
    print(f"   [{'HIT ' if ok else 'miss'}] {p:32s} -> {body.strip()[-60:]!r}  ({dt:.0f}s)")
print(f"   => {hits}/{len(FACTS)}   (baseline b1024: 4/4; healthy: 4/4)")

print("### loop metric (n=200)")
body, dt = run("Write a detailed tutorial explaining how login auditing works and why it matters.", 200)
w = body.split()
u = len(set(w)) / len(w) if w else 0.0
g6 = [" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
rep6 = max((g6.count(g) for g in set(g6)), default=0)
print(f"   uniq={u:5.2f} rep6={rep6:3d} words={len(w):4d}  ({dt:.0f}s)")
print(f"   head: {body.strip()[:110]!r}")
print(f"   [b1024 was uniq 0.24 rep6 16; healthy uniq 0.72 rep6 1]")
