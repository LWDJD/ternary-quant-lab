"""Confirm the packed 35B MoE's folded set on the remote."""
import sys

sys.path.insert(0, "/mnt/workspace/prismwork/prism-src/gguf-py")
from gguf import GGUFReader

r = GGUFReader("/tmp/moe35-ptq10.gguf")
names = [str(s) for s in r.fields["prism.hadamard.weight_names"].contents()]
print(f"folded        = {len(names)}")
print(f"ssm_out       = {sum('ssm_out' in x for x in names)}")
print(f"attn_gate     = {sum('attn_gate' in x for x in names)}")
print(f"attn_qkv      = {sum('attn_qkv' in x for x in names)}")
print(f"ffn_down_exps = {sum('ffn_down_exps' in x for x in names)}")
print(f"output.weight = {sum(x == 'output.weight' for x in names)}")
print(f"gdn_v_grouped = {r.fields['prism.hadamard.gdn_v_grouped'].contents()}")
print(f"block_size    = {r.fields['prism.hadamard.block_size'].contents()}")
print(f"file_type     = {r.fields['general.file_type'].contents()}")
