"""Which local GGUFs can actually be folded?  Only arches in the fork's
_HADAMARD_ARCHS set (LLAMA, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE, QWEN3NEXT) are
accepted, so check the architecture of every local model before picking an
iteration testbed.
"""
import glob
import os
import sys

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")
from gguf import GGUFReader

ROOT = r"D:\AI\lmstudio\models"
OK = {"llama", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next"}

for p in sorted(glob.glob(os.path.join(ROOT, "**", "*.gguf"), recursive=True)):
    try:
        r = GGUFReader(p)
        arch = str(r.fields["general.architecture"].contents()) if "general.architecture" in r.fields else "?"
        name = str(r.fields["general.name"].contents())[:34] if "general.name" in r.fields else ""
        n = len(r.tensors)
        flag = "FOLD-OK " if arch in OK else "        "
        print(f"  {flag} {os.path.getsize(p)/2**20:8.0f} MiB  arch={arch:12s} tensors={n:5d}  {name}")
    except Exception as e:
        print(f"           {os.path.basename(p):52s} <{type(e).__name__}>")
