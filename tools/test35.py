"""Controlled comparison: same harness, three models, one variable at a time.

The previous run changed TWO things at once (folded ssm_out AND switched the
scale rule to --density 0.672), so it cannot say which one broke it.  The old
artefact is still on disk, so re-test it under identical conditions.

Also fixes the fact probe: this model emits a long "thinking process" preamble,
so 32 tokens never reach the answer.  Use 300 tokens and search the whole body.
"""
import os
import re
import subprocess
import time

B = r"D:\Project\openhanako\workbench\prism-tip\build-vs\bin\Release"
NEW = r"D:\Project\openhanako\workbench\prism-pack\q\moe35-ptq10.gguf"
DENS = NEW
LS = r"D:\Project\openhanako\workbench\prism-pack\q\moe35-ls.gguf"
OLD = r"D:\Project\openhanako\workbench\prism-pack\q\qwen36-moe-b1024.gguf"
TGT = (r"D:\AI\lmstudio\models\HauhauCS\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
       r"\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf")
L = r"D:\Project\openhanako\workbench\prism-pack\logs35"
os.makedirs(L, exist_ok=True)

FACTS = [
    ("france", "The capital of France is", "Paris"),
    ("japan", "The capital of Japan is", "Tokyo"),
    ("germany", "The capital of Germany is", "Berlin"),
    ("water", "Water is made of hydrogen and", "oxygen"),
]

MODELS = [("healthy", TGT), ("old", OLD), ("dens", DENS), ("ls", LS)]


def run(label, model, prompt, extra, ntok, tag=""):
    safe = re.sub(r"[^a-z0-9]+", "_", f"{label}{tag}")
    out = os.path.join(L, safe + ".txt")
    args = ["-m", model, "-ngl", "99", "-c", "8192", "-p", prompt, "-n", str(ntok),
            "--temp", "0", "--seed", "1", "--jinja", "-st", "--no-display-prompt"] + extra
    t0 = time.time()
    with open(out, "wb") as fo, open(out + ".err", "wb") as fe:
        subprocess.run([os.path.join(B, "llama-completion.exe")] + args,
                       stdout=fo, stderr=fe, timeout=3600)
    txt = open(out, "rb").read().decode("utf-8", "replace").strip()
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


print("### facts (n=300 so a thinking preamble can finish)")
for label, model in MODELS:
    hits = 0
    for pid, p, want in FACTS:
        body, dt = run(label, model, p, [], 300, tag=f"-{pid}")
        ok = want.lower() in body.lower()
        hits += ok
        print(f"   {label:8s} [{'HIT ' if ok else 'miss'}] {p:32s} -> {body[-60:]!r}")
    print(f"   => {label}: {hits}/{len(FACTS)}\n")

print("### loop metric (400 tokens, greedy)")
PROMPT = "Write a detailed tutorial explaining how login auditing works and why it matters."
for label, model in MODELS:
    body, dt = run(label, model, PROMPT, [], 400, tag="-loop")
    u, r6, rn = metrics(body)
    print(f"   {label:8s} uniq={u:5.2f} rep6={r6:4d} run={rn:3d} words={len(body.split()):4d} {dt:.0f}s")
    print(f"      head: {body[:100]!r}")
