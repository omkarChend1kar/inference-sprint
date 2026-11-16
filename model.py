import json
from types import SimpleNamespace
import torch
import torch.nn as nn
from safetensors.torch import load_file
from layer import LayerCache, RMSNorm, DecoderLayer, rope_tables


class Qwen3(nn.Module):
    def __init__(self, cfg, max_pos=4096):
        super().__init__()

        # embed_tokens: nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # layers: nn.ModuleList of cfg.num_hidden_layers DecoderLayer(cfg)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        # norm: RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        cos, sin = rope_tables(cfg.head_dim, max_pos, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, caches=None, start_index=0):                         # ids: (B, S)
        # 1. x = embedding lookup of ids             → (B, S, 1024)
        x = self.embed_tokens(ids)
        # 2. for each layer: x = layer(x, self.cos, self.sin)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, self.cos, self.sin, cache=cache, start_index=start_index)
        # 3. x = final norm of x
        x = self.norm(x)
        # 4. logits = x @ self.embed_tokens.weight.T  → (B, S, 151936)
        logits = x @ self.embed_tokens.weight.T
        # return logits
        return logits


def load(path):
    cfg = SimpleNamespace(**json.load(open(f"{path}/config.json")))
    model = Qwen3(cfg)
    state = load_file(f"{path}/model.safetensors")
    state = {k.removeprefix("model."): v.float()
             for k, v in state.items() if k != "lm_head.weight"}
    model.load_state_dict(state, strict=True)
    return model.eval()

@torch.no_grad()
def generate(model, ids, max_new_tokens, eos_id=None):
    for _ in range(max_new_tokens):
        # 1. logits = model(ids)
        logits = model(ids)
        # 2. next_id = logits[:, -1].argmax(dim=-1, keepdim=True)     → (1, 1)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        # 3. ids = torch.cat([ids, next_id], dim=1)
        ids = torch.cat([ids, next_id], dim=1)
        # 4. if eos_id is not None and next_id.item() == eos_id: break
        if eos_id is not None and next_id.item() == eos_id:
            break
    return ids

@torch.no_grad()
def generate_cached(model, ids, max_new_tokens, eos_id=None):
    caches = [LayerCache() for _ in model.layers]      # 28 empty caches

    # PREFILL — whole prompt once, positions start at 0, fills every cache

    # 1. logits = model(ids, caches, start=0)
    logits = model(ids, caches, start_index=0)
    # 2. next_id = argmax of the last row             → (1, 1)
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    # 3. out = torch.cat([ids, next_id], dim=1)
    out = torch.cat([ids, next_id], dim=1)
    # 4. pos = ids.shape[1]                            ← position of next_id
    pos = ids.shape[1]

    # DECODE — one token per pass
    # repeat max_new_tokens - 1 times:
    #     stop if next_id is eos_id
    #     logits = model(next_id, caches, start=pos)  ← only ONE token in
    #     pos += 1
    #     next_id = argmax of the last row
    #     out = torch.cat([out, next_id], dim=1)
    # return out
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and next_id.item() == eos_id:
            break
        logits = model(next_id, caches, start_index=pos)
        pos += 1
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
    return out