"""Acceptance: facts + repetition + speed, for the ref-rule MoE pack.

Judged with the criteria already fixed for this project (all greedy, --jinja, -st):
  facts     >= 3/4
  rep6      <= 3          (healthy is 1; the old pack sat at 8-11)
  uniq      ~ 0.72        (healthy)
"""
from __future__ import annotations

import os
import re
import subprocess
import time

import sys

BIN = sys.argv[2] if len(sys.argv) > 2 else "/tmp/llb/bin/llama-cli"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref.gguf"
RP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
FLAGS = ["-c", "4096", "-t", "8", "-ngl", "0", "--jinja", "-st",
         "--no-display-prompt", "--temp", "0"]
if RP:
    FLAGS += ["--repeat-penalty", str(RP)]

FACTS = [
    ("The capital of France is", "Paris"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
]


def run(prompt: str, n: int = 300) -> str:
    cmd = [BIN, "-m", MODEL, "-p", prompt, "-n", str(n), *FLAGS]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    dt = time.time() - t0
    out = (r.stdout or "")
    # the loader spinner uses backspaces; drop control chars and its own line first
    out = out.replace("\x08", "\n")
    out = re.sub(r"[\x00-\x08\x0b-\x1f]", "", out)
    out = re.sub(r"Loading model\.\.\.?", " ", out)
    out = re.sub(r"\[ Prompt:.*?Exiting\.\.\.", " ", out, flags=re.S)
    if "</think>" in out:
        out = out.split("</think>")[-1]
    out = out.replace(prompt, " ")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    print(f"    ({dt:.0f}s)")
    return " ".join(lines)


def metrics(text: str):
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0, 0, 0
    uniq = len(set(w.lower() for w in words)) / len(words)
    grams = [tuple(w.lower() for w in words[i:i + 6]) for i in range(len(words) - 5)]
    rep6 = 0
    if grams:
        counts: dict[tuple, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        rep6 = max(counts.values())
    return uniq, rep6, len(words)


print(f"model: {MODEL}  ({os.path.getsize(MODEL) / 2**30:.2f} GiB)")
print(f"repeat-penalty: {RP if RP else 'none'}")
print(f"note  : --scale ref --theta 0.38  (d = 1.40*mean|x|, rank-86 equivalent)\n")
print("=== facts ===")
hits = 0
for prompt, expect in FACTS:
    out = run(prompt)
    # word-boundary match: a bare substring let "Au" pass on "because"
    ok = re.search(r"\b" + re.escape(expect) + r"\b", out, re.I) is not None
    hits += ok
    tail = out[-160:].replace("\n", " ")
    print(f"  [{'HIT ' if ok else 'miss'}] {prompt:46s} -> {tail!r}")
print(f"  => {hits}/{len(FACTS)}")

print("\n=== generation (repetition) ===")
text = run("Write a short tutorial explaining how login auditing works and why it matters.", 300)
uniq, rep6, nwords = metrics(text)
print(f"  uniq={uniq:.2f}  rep6={rep6}  words={nwords}   [healthy 0.72 / 1]")
print(f"  head={text[:150]!r}")

print("\n=== verdict ===")
print(f"  facts {hits}/4   (need >=3)      {'PASS' if hits >= 3 else 'FAIL'}")
print(f"  rep6  {rep6}     (need <=3)      {'PASS' if rep6 <= 3 else 'FAIL'}")
print(f"  uniq  {uniq:.2f}  (want ~0.72)    "
      f"{'PASS' if uniq > 0.55 else 'FAIL'}")
