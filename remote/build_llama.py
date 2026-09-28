"""Configure + build the fork CPU-only, in the background (caller polls).

Build dir goes on /tmp (local SSD); the source lives on the NAS.  64 cores here,
so a from-scratch build should be a few minutes rather than the 20+ it took on
the 8-core instance.
"""
import os
import subprocess
import time

SRC = "/mnt/workspace/prismwork/prism-src"
BLD = "/tmp/llb"
LOG = "/mnt/workspace/prismwork/build.log"

CFG = (
    "cmake -S {src} -B {bld} -DCMAKE_BUILD_TYPE=Release "
    "-DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF "
    "-DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DLLAMA_CURL=OFF "
    "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF"
).format(src=SRC, bld=BLD)

script = f"""set -e
echo "== configure $(date +%H:%M:%S)"
{CFG}
echo "== build $(date +%H:%M:%S)"
cmake --build {BLD} -j 64
echo "== done $(date +%H:%M:%S)"
ls -la {BLD}/bin/ | head -20
"""
open("/tmp/_build.sh", "w").write(script)
p = subprocess.Popen(["bash", "/tmp/_build.sh"], stdout=open(LOG, "wb"),
                     stderr=subprocess.STDOUT, start_new_session=True)
print("build started pid", p.pid)
time.sleep(25)
r = subprocess.run(f"tail -8 {LOG}", shell=True, capture_output=True, text=True)
print(r.stdout + r.stderr)
