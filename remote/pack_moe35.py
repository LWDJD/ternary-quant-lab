"""Wait for the MoE download, then pack it with the corrected recipe.

Recipe is chosen to REMOVE EVERY DEVIATION from the reference (Bonsai 2):
  block 1024          same as the reference and the whitepaper
  --density 0.672     reference keeps a constant 67.2% non-zero trits; the
                      least-squares scale gave 50.6%
  --fallback-quant    ffn_down_exps has input dim 512, so at block 1024 it cannot
                      be rotated; Q4_0 beats unrotated ternary and F16 is huge

The one deviation left is the sign vector: ours is deterministic-random, the
reference stores explicit values we cannot recover.
"""
import os
import subprocess
import sys
import time

SRC = "/mnt/workspace/prismwork/prism-src"
HF = "/tmp/moe35"
OUT = "/tmp/moe35-ptq10.gguf"
WORK = "/mnt/workspace/prismwork"


def sh(cmd, timeout=7200):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()


print("=== wait for download ===", flush=True)
for i in range(360):
    rc, out = sh("pgrep -fc '[m]odelscope download' || true")
    running = out.strip() not in ("", "0")
    shards = sh(f"ls {HF}/*.safetensors 2>/dev/null | wc -l")[1].strip()
    inc = sh(f"ls {HF}/*.incomplete 2>/dev/null | wc -l")[1].strip()
    size = sh(f"du -sh {HF} 2>/dev/null | cut -f1")[1].strip()
    print(f"  [{i:3d}] running={running} shards={shards} incomplete={inc} size={size}", flush=True)
    if not running and inc == "0" and shards == "26":
        break
    time.sleep(30)
else:
    print("  TIMEOUT waiting for download")
    sys.exit(1)
print("  download complete", flush=True)

print("\n=== pack 35B MoE (block 1024, density 0.672) ===", flush=True)
t0 = time.time()
rc, out = sh(f"cd {WORK} && {sys.executable} -u prism_pack_hf.py {HF} {OUT} "
             f"--block 1024 --density 0.672 --jobs 8 --fallback-quant Q4_0 "
             f"--fork {SRC} 2>&1 | tail -12", timeout=10800)
print(out[-2500:], flush=True)
print(f"  rc={rc}  elapsed={time.time()-t0:.0f}s", flush=True)

print("\n=== verify ===", flush=True)
sys.path.insert(0, f"{SRC}/gguf-py")
try:
    from gguf import GGUFReader
    r = GGUFReader(OUT)
    names = [str(s) for s in r.fields["prism.hadamard.weight_names"].contents()]
    print(f"  file={OUT}  {os.path.getsize(OUT)/2**30:.2f} GiB")
    print(f"  folded = {len(names)}")
    print(f"  ssm_out folded = {sum('ssm_out' in n for n in names)}")
    print(f"  attn_gate folded = {sum('attn_gate' in n for n in names)}")
    print(f"  gdn_v_grouped = {r.fields['prism.hadamard.gdn_v_grouped'].contents() if 'prism.hadamard.gdn_v_grouped' in r.fields else '<ABSENT>'}")
    print(f"  block_size = {r.fields['prism.hadamard.block_size'].contents()}")
    print(f"  file_type = {r.fields['general.file_type'].contents()}")
except Exception as e:
    print(f"  verify failed: {type(e).__name__} {e}")

print("\n=== sha256 ===", flush=True)
print(sh(f"sha256sum {OUT}")[1])
print("\nDONE", flush=True)
