"""Free NAS quota and fetch the dense Qwen3.8-27B.

The 67 GB MoE model was our own download and is re-creatable, so it goes; the
dense 27B (55.6 GB) replaces it.  Only touches /mnt/workspace/prismwork.
"""
import os
import subprocess


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


M = "/mnt/workspace/prismwork/models"
print("=== before ===")
print(sh(f"du -sh {M}/* 2>/dev/null; df -h /mnt/workspace | tail -1"))

old = f"{M}/qwen36-35b-a3b"
if os.path.exists(old):
    sz = sh(f"du -sh {old} 2>/dev/null").split()[0]
    sh(f"rm -rf {old}")
    print(f"  removed {old}  ({sz})")

print("\n=== after cleanup ===")
print(sh(f"du -sh {M}/* 2>/dev/null; ls {M} 2>/dev/null"))

# start the download in the background so the caller can poll
os.makedirs(M, exist_ok=True)
log = "/mnt/workspace/prismwork/dl27b.log"
cmd = (f"cd {M} && modelscope download --model Qwen/Qwen3.8-27B "
       f"--local_dir qwen38-27b > {log} 2>&1")
p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
print(f"\n  download started pid={p.pid}  log={log}")
