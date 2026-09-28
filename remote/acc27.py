import os
import subprocess
import time

B = "/tmp/llb/bin/llama-completion"
M = "/tmp/rob.gguf"
F = [("france", "The capital of France is", "Paris"),
     ("water", "Water is made of hydrogen and", "oxygen")]


def run(p, n):
    a = [B, "-m", M, "-ngl", "0", "-c", "4096", "-p", p, "-n", str(n), "--temp", "0",
         "--seed", "1", "--jinja", "-st", "--no-display-prompt", "--threads", "8"]
    t = time.time()
    r = subprocess.run(a, capture_output=True, text=True, timeout=3000, errors="replace")
    return (r.stdout or ""), time.time() - t


print(f"file: {os.path.getsize(M)/2**30:.2f} GiB  (--scale robust)")
h = 0
for pid, p, w in F:
    b, dt = run(p, 260)
    ok = w.lower() in b.lower()
    h += ok
    print(f"  [{'HIT ' if ok else 'miss'}] {p:30s} -> {b.strip()[-58:]!r} ({dt:.0f}s)")
print(f"  => {h}/{len(F)}")

b, dt = run("Write a detailed tutorial explaining how login auditing works and why it matters.", 200)
w = b.split()
u = len(set(w)) / len(w) if w else 0
g = [" ".join(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
r6 = max((g.count(x) for x in set(g)), default=0)
print(f"  loop uniq={u:.2f} rep6={r6} words={len(w)} ({dt:.0f}s)  [healthy 0.72/1]")
print(f"  head={b.strip()[:88]!r}")
