"""Re-fetch the 35B MoE (the main target) to /tmp.

/tmp is instance-local (335 GB, no NAS quota) -- right choice for a one-shot pack,
since /mnt/workspace has a ~100 GB quota the 70 GB model would eat.
"""
import glob
import os
import subprocess
import time

DST = "/tmp/moe35"
LOG = "/tmp/moe35-dl.log"


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return ((r.stdout or "") + (r.stderr or "")).strip()


print("=== probe candidate repos ===")
for mid in ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B-Instruct",
            "Qwen/Qwen3.6-35B-A3B-Base"):
    ns, name = mid.split("/")
    out = sh(f"timeout 60 curl -s -o /dev/null -w '%{{http_code}}' "
             f"https://www.modelscope.cn/api/v1/models/{ns}/{name}")
    print(f"  {mid:34s} HTTP {out}")

print("\n=== disk ===")
print(sh("df -h /tmp | tail -1"))

print("\n=== existing download running? ===")
print(sh("ps aux | grep -c '[m]odelscope download'"))

print(f"\n=== start download -> {DST} ===")
os.makedirs(DST, exist_ok=True)
cmd = (f"cd /tmp && modelscope download --model Qwen/Qwen3.6-35B-A3B "
       f"--local_dir {DST} > {LOG} 2>&1")
p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
print(f"  pid={p.pid}  log={LOG}")
time.sleep(20)
print("\n=== first log lines ===")
print(sh(f"tail -c 1200 {LOG}"))
