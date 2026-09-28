"""Short decisive test: the ls variant only.

old  (no ssm_out fold, LS scale) already measured 4/4 facts, uniq 0.24.
ls   (ssm_out folded,      LS scale) differs from old by exactly one variable.
So only ls needs running -- the comparison baseline already exists.
"""
import os
import re
import subprocess
import time

B = r"D:\Project\openhanako\workbench\prism-tip\build-vs\bin\Release"
LS = r"D:\Project\openhanako\workbench\prism-pack\q\moe35-ls.gguf"
L = r"D:\Project\openhanako\workbench\prism-pack\logs35"
os.makedirs(L, exist_ok=True)

FACTS = [
    ("france", "The capital of France is", "Paris"),
    ("japan", "The capital of Japan is", "Tokyo"),
    ("germany", "The capital of Germany is", "Berlin"),
    ("water", "Water is made of hydrogen and", "oxygen"),
]


def run(prompt, ntok, tag):
    out = os.path.join(L, f"quick-{tag}.txt")
    args = ["-m", LS, "-ngl", "99", "-c", "8192", "-p", prompt, "-n", str(ntok),
            "--temp", "0", "--seed", "1", "--jinja", "-st", "--no-display-prompt"]
    t0 = time.time()
    with open(out, "wb") as fo, open(out + ".err", "wb") as fe:
        subprocess.run([os.path.join(B, "llama-completion.exe")] + args,
                       stdout=fo, stderr=fe, timeout=3600)
    return open(out, "rb").read().decode("utf-8", "replace").strip(), time.time() - t0


print("### ls variant -- facts (n=300)")
hits = 0
for pid, p, want in FACTS:
    body, dt = run(p, 300, pid)
    ok = want.lower() in body.lower()
    hits += ok
    print(f"   [{'HIT ' if ok else 'miss'}] {p:32s} -> {body[-64:]!r}  ({dt:.0f}s)")
print(f"   => {hits}/{len(FACTS)}   (old was 4/4, healthy 4/4)")

print("\n### ls variant -- loop metric (n=300)")
body, dt = run("Write a detailed tutorial explaining how login auditing works and why it matters.",
               300, "loop")
w = body.split()
u = len(set(w)) / len(w) if w else 0.0
g6 = [" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
rep6 = max((g6.count(g) for g in set(g6)), default=0)
print(f"   uniq={u:5.2f} rep6={rep6:3d} words={len(w):4d}  ({dt:.0f}s)   "
      f"[old was uniq 0.24 rep6 16; healthy uniq 0.72 rep6 1]")
print(f"   head: {body[:120]!r}")
