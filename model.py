import json
from types import SimpleNamespace
import torch
import torch.nn as nn
from safetensors.torch import load_file
from layer import RMSNorm, DecoderLayer, rope_tables


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

    def forward(self, ids):                         # ids: (B, S)
        # 1. x = embedding lookup of ids             → (B, S, 1024)
        x = self.embed_tokens(ids)
        # 2. for each layer: x = layer(x, self.cos, self.sin)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
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