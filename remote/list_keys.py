"""List the tensor names in the Qwen3.8-27B checkpoint, so the fold can find them."""
from __future__ import annotations

import json

from safetensors import safe_open

HF = "/tmp/q38-27b"
with safe_open(f"{HF}/model-00001-of-00018.safetensors", framework="pt") as f:
    keys = sorted(f.keys())
print(f"shard 1 keys: {len(keys)}")
for k in keys[:20]:
    print(f"  {k}")

print("\nlayer 0 (all):")
for k in keys:
    if k.startswith("model.layers.0."):
        print(f"  {k}")

try:
    cfg = json.load(open(f"{HF}/config.json"))
    for key in ("model_type", "architectures", "num_hidden_layers", "hidden_size",
                "tie_word_embeddings", "full_attention_interval", "linear_attention"):
        if key in cfg:
            print(f"config.{key} = {cfg[key]}")
    for key in cfg:
        if "attention" in key or "linear" in key:
            print(f"config.{key} = {cfg[key]}")
except Exception as e:  # noqa: BLE001
    print(f"config: {e}")
