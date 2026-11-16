import torch
from safetensors import safe_open
f = safe_open("../models/Qwen3-0.6B/model.safetensors", "pt")
keys = list(f.keys())
print(len(keys), "tensors")
for k in keys:
    if "layers." not in k or "layers.0." in k:
        print(k, tuple(f.get_slice(k).get_shape()))


for name in ["model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.self_attn.v_proj.weight",
             "model.norm.weight"]:
    print(name, name in keys)

print("tensors in layer 0:", sum(k.startswith("model.layers.0.") for k in keys))


import json, torch
cfg = json.load(open("../models/Qwen3-0.6B/config.json"))
print("tie_word_embeddings:", cfg.get("tie_word_embeddings"))

E  = f.get_tensor("model.embed_tokens.weight")
LM = f.get_tensor("lm_head.weight")
print("identical:", torch.equal(E, LM))