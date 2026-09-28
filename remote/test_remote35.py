"""Judge both 35B variants on the remote CPU build -- no download needed.

The remote already has the CPU-only build from the same revision (adfffbe), and
that build has PTQ1_0 vec_dot for AVX-512/AVX2.  64 cores.  So the two packs can
be compared directly, which isolates the one variable that changed between them:
the scale rule.

  moe35-ls.gguf      ssm_out folded, least-squares scale
  moe35-ptq10.gguf   ssm_out folded, --density 0.672   (this one output digit soup)

Baseline from the identical harness on the local Vulkan build:
  healthy IQ2_M   facts 4/4   uniq 0.72  rep6 1
"""
import os
import re
import subprocess
import time

BIN = "/tmp/llb/bin/llama-completion"
MODELS = [("ls", "/tmp/moe35-ls.gguf"), ("dens", "/tmp/moe35-ptq10.gguf")]

FACTS = [
    ("france", "The capital of France is", "Paris"),
    ("japan", "The capital of Japan is", "Tokyo"),
    ("germany", "The capital of Germany is", "Berlin"),
    ("water", "Water is made of hydrogen and", "oxygen"),
]

if not os.path.exists(BIN):
    print(f"MISSING {BIN}")
    raise SystemExit(1)


def run(model, prompt, ntok):
    args = [BIN, "-m", model, "-ngl", "0", "-c", "8192", "-p", prompt, "-n", str(ntok),
            "--temp", "0", "--seed", "1", "--jinja", "-st", "--no-display-prompt",
            "--threads", "32"]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True, timeout=5400)
    txt = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    return txt, time.time() - t0


def metrics(text):
    w = text.split()
    if not w:
        return 0.0, 0, 0
    g6 = [" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
    rep6 = max((g6.count(g) for g in set(g6)), default=0)
    g5 = [" ".join(w[i:i + 5]) for i in range(max(0, len(w) - 4))]
    rl = best = 0
    for i in range(1, len(g5)):
        rl = rl + 1 if g5[i] == g5[i - 1] else 0
        best = max(best, rl)
    return len(set(w)) / len(w), rep6, best


print("### facts (n=300)")
for label, model in MODELS:
    if not os.path.exists(model):
        print(f"   {label}: MISSING {model}")
        continue
    hits = 0
    for pid, p, want in FACTS:
        body, dt = run(model, p, 300)
        ok = want.lower() in body.lower()
        hits += ok
        print(f"   {label:5s} [{'HIT ' if ok else 'miss'}] {p:32s} -> {body[-70:]!r}  ({dt:.0f}s)")
    print(f"   => {label}: {hits}/{len(FACTS)}\n")

print("### loop metric (n=400)")
PROMPT = "Write a detailed tutorial explaining how login auditing works and why it matters."
for label, model in MODELS:
    if not os.path.exists(model):
        continue
    body, dt = run(model, PROMPT, 400)
    u, r6, rn = metrics(body)
    print(f"   {label:5s} uniq={u:5.2f} rep6={r6:4d} run={rn:3d} words={len(body.split()):4d} {dt:.0f}s")
    print(f"      head: {body[:100]!r}")
