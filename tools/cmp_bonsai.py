"""Ground truth: compare Bonsai 2's folded-tensor list against mine.

Bonsai 2 works.  If its prism.hadamard.weight_names excludes attn_gate and
output.weight, that confirms both the diagnosis and the exact target set --
including whether those tensors are still stored as PTQ1_0.
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")
from gguf import GGUFReader

REF = r"D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
MINE = r"D:\Project\openhanako\workbench\prism-pack\q\qwen38-27b-ptq10.gguf"


def kind(n):
    if n.startswith("blk."):
        parts = n.split(".", 2)
        return parts[2] if len(parts) > 2 else n
    return n


def load(path, tag):
    r = GGUFReader(path)
    print(f"### {tag}: {os.path.basename(path)}  ({os.path.getsize(path)/2**30:.2f} GiB)")
    if "prism.hadamard.weight_names" not in r.fields:
        print("   no hadamard metadata")
        return None, r
    names = sorted(str(s) for s in r.fields["prism.hadamard.weight_names"].contents())
    print(f"   folded = {len(names)}")
    bs = r.fields.get("prism.hadamard.block_size")
    print(f"   block_size = {bs.contents() if bs else '?'}")
    print(f"   sign_widths = {list(r.fields['prism.hadamard.sign_widths'].contents()) if 'prism.hadamard.sign_widths' in r.fields else '?'}")
    hist = Counter(kind(n) for n in names)
    print(f"   kinds: {dict(sorted(hist.items()))}")
    # what is NOT folded but is PTQ1_0?
    ty = Counter()
    for t in r.tensors:
        if str(t.tensor_type).endswith("PTQ1_0"):
            ty[t.name] += 1
    ptq_unfolded = sorted(kind(n) for n in ty if n not in set(names))
    print(f"   PTQ1_0 total = {len(ty)}   PTQ1_0-but-NOT-folded kinds = {dict(Counter(ptq_unfolded))}")
    return set(names), r


nb, _ = load(REF, "BONSAI 2 (works)")
print()
nm, _ = load(MINE, "MINE (broken)")

if nb is not None and nm is not None:
    print("\n=== diff ===")
    print(f"  only in Bonsai: {sorted(nb - nm)[:8]}{' ...' if len(nb-nm) > 8 else ''}  (n={len(nb-nm)})")
    print(f"  only in mine  : {sorted(nm - nb)[:8]}{' ...' if len(nm-nb) > 8 else ''}  (n={len(nm-nb)})")
    print("\n=== the suspects ===")
    for tag, s in (("Bonsai", nb), ("mine", nm)):
        print(f"  {tag}: attn_gate={sum('attn_gate' in n for n in s)}  "
              f"output.weight={'output.weight' in s}  "
              f"ssm_out={sum('ssm_out' in n for n in s)}")
