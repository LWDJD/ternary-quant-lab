"""Rebuild the tiny qwen35 with CORRECT GDN shapes, then run the rotator test.

Shapes taken from the real Qwen3.8-27B safetensors headers, not guessed:
    conv1d.weight   [q_dim + k_dim + v_dim, 1, conv_kernel]
    in_proj_qkv     [q_dim + k_dim + v_dim, hidden]
    in_proj_z       [num_v_heads * head_v_dim, hidden]
    in_proj_a/b     [num_v_heads, hidden]
    out_proj        [hidden, num_v_heads * head_v_dim]
    A_log / dt_bias [num_v_heads]
    norm            [head_v_dim]
The previous attempt used only the v-width for conv1d, which produced a
zero-size tensor during the V-head reorder.
"""
import json
import os
import shutil

import torch
from safetensors.torch import save_file

OUT = "/tmp/q35"
SRC = "/mnt/workspace/prismwork/models/qwen38-27b"
H = 128
HEADS, KV, HD = 4, 2, 32
LKH, LVH, LHD = 2, 4, 32
CONV = 4
LAYERS = 4
INTER = 256

QKV = LKH * LHD * 2 + LVH * LHD      # 2*2*32 + 4*32 = 256
VW = LVH * LHD                       # 128

layer_types = ["full_attention" if i % 2 == 0 else "linear_attention" for i in range(LAYERS)]

cfg = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": H,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": HD,
        "linear_num_key_heads": LKH,
        "linear_num_value_heads": LVH,
        "linear_key_head_dim": LHD,
        "linear_value_head_dim": LHD,
        "linear_conv_kernel_dim": CONV,
        "full_attention_interval": 2,
        "layer_types": layer_types,
        "vocab_size": 1024,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    },
    "tie_word_embeddings": False,
}

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

g = torch.Generator().manual_seed(7)


def r(*shape):
    return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)


sd = {
    "model.language_model.embed_tokens.weight": r(1024, H),
    "model.language_model.norm.weight": r(H),
    "lm_head.weight": r(1024, H),
}
for i, lt in enumerate(layer_types):
    p = f"model.language_model.layers.{i}"
    sd[f"{p}.input_layernorm.weight"] = r(H)
    sd[f"{p}.post_attention_layernorm.weight"] = r(H)
    if lt == "linear_attention":
        q = f"{p}.linear_attn"
        sd[f"{q}.A_log"] = r(LVH)
        sd[f"{q}.dt_bias"] = r(LVH)
        sd[f"{q}.conv1d.weight"] = r(QKV, 1, CONV)          # <- the fix
        sd[f"{q}.in_proj_qkv.weight"] = r(QKV, H)
        sd[f"{q}.in_proj_z.weight"] = r(VW, H)
        sd[f"{q}.in_proj_a.weight"] = r(LVH, H)
        sd[f"{q}.in_proj_b.weight"] = r(LVH, H)
        sd[f"{q}.out_proj.weight"] = r(H, VW)
        sd[f"{q}.norm.weight"] = r(LHD)
    else:
        q = f"{p}.self_attn"
        sd[f"{q}.q_proj.weight"] = r(HEADS * HD, H)
        sd[f"{q}.k_proj.weight"] = r(KV * HD, H)
        sd[f"{q}.v_proj.weight"] = r(KV * HD, H)
        sd[f"{q}.o_proj.weight"] = r(H, HEADS * HD)
        sd[f"{q}.q_norm.weight"] = r(HD)
        sd[f"{q}.k_norm.weight"] = r(HD)
    m = f"{p}.mlp"
    sd[f"{m}.gate_proj.weight"] = r(INTER, H)
    sd[f"{m}.up_proj.weight"] = r(INTER, H)
    sd[f"{m}.down_proj.weight"] = r(H, INTER)

save_file(sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "generation_config.json"):
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, f))

print(f"  rebuild done: {len(sd)} tensors, layers={layer_types}")
print(f"  qkv={QKV} vw={VW} conv1d=[{QKV},1,{CONV}]")
