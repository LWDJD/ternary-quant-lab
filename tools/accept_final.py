"""Final acceptance run on the deliverable, with the recommended settings.

Model: prism-pack/q/qwen36-moe-b1024.gguf (block 1024, least-squares scale,
ffn_down_exps via Q4_0) -- facts 4/4 without any penalty, but it loops.

This measures both states so the recommendation is grounded:
  A  --jinja -st, greedy                        (raw behaviour)
  B  --jinja -st, greedy + --repeat-penalty 1.3 (recommended)
"""
import os
import re
import subprocess
import time

B = r"D:\Project\openhanako\workbench\prism-tip\build-vs\bin\Release"
M = r"D:\Project\openhanako\workbench\prism-pack\q\qwen36-moe-b1024.gguf"
L = r"D:\Project\openhanako\workbench\prism-pack\logs35"
os.makedirs(L, exist_ok=True)

FACTS = [("france", "The capital of France is", "Paris"),
         ("water", "Water is made of hydrogen and", "oxygen")]
CASES = [("A raw", []), ("B +rpen", ["--repeat-penalty", "1.3"])]
LOOP = "Write a detailed tutorial explaining how login auditing works and why it matters."


def run(prompt, ntok, extra, tag):
    out = os.path.join(L, f"acc-{tag}.txt")
    args = ["-m", M, "-ngl", "99", "-c", "8192", "-p", prompt, "-n", str(ntok),
            "--temp", "0", "--seed", "1", "--jinja", "-st", "--no-display-prompt"] + extra
    t0 = time.time()
    with open(out, "wb") as fo, open(out + ".err", "wb") as fe:
        subprocess.run([os.path.join(B, "llama-completion.exe")] + args,
                       stdout=fo, stderr=fe, timeout=3600)
    return open(out, "rb").read().decode("utf-8", "replace").strip(), time.time() - t0


def metrics(text):
    w = text.split()
    if not w:
        return 0.0, 0
    g6 = [" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
    return len(set(w)) / len(w), max((g6.count(g) for g in set(g6)), default=0)


print(f"model: {os.path.basename(M)}  {os.path.getsize(M)/2**30:.2f} GiB")
for label, extra in CASES:
    hits = 0
    for pid, p, want in FACTS:
        body, dt = run(p, 300, extra, f"{label.strip()}-{pid}")
        ok = want.lower() in body.lower()
        hits += ok
    body, _ = run(LOOP, 300, extra, f"{label.strip()}-loop")
    u, r6 = metrics(body)
    print(f"  {label:7s} facts={hits}/{len(FACTS)}  uniq={u:5.2f}  rep6={r6:3d}  "
          f"head={body[:60]!r}")
print("\nreference: healthy IQ2_M = facts 4/4, uniq 0.72, rep6 1")
