import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # your code here
        in_dtype = x.dtype # this is to record what came in (float32 CPU, bfloat16 on GPU)
        x = x.to(torch.float32)

        mean_sq = x.pow(2).mean(-1, keepdim=True) # average of x squared, last axis, keepdim=True
        x = x * torch.rsqrt(mean_sq + self.eps) # x times rsqrt(mean_sq + self.eps)

        return x.to(in_dtype) * self.weight 


def rope_tables(head_dim, max_pos, theta=1_000_000.0):
    # returns cos, sin — each (max_pos, head_dim)
    # 64 distinct frequencies, radians per token, per pair
    # basically, 64 distinct sinusoids (rulers), each with a different frequency, that we can use to encode position information into the model's attention mechanism.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    #
    pos = torch.arange(max_pos).float()

    #
    angles = torch.outer(pos, inv_freq)

    #
    emb = torch.cat([angles, angles], dim=-1)

    #
    return torch.cos(emb), torch.sin(emb)


def apply_rope(x, cos, sin):
    # x: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (rigtness / leftness of a pair at that position, up / downness of a pair at that position)
    # returns: (batch, seq_len, num_heads, head_dim)
    # your code here
    seq_len = x.shape[-2]

    # group the last dimension into pairs of numbers (64 pairs for 128 channels)
    half = x.shape[-1] // 2

    # take rows for this sentence, add batch & head axes
    cos = cos[:seq_len][None, None, :, :half]
    sin = sin[:seq_len][None, None, :, :half]

    a = x[..., :half]    # each pair's first number  (channels 0..63)
    b = x[..., half:]    # each pair's second number (channels 64..127)

    new_a = a * cos - b * sin
    new_b = a * sin + b * cos

    return torch.cat([new_a, new_b], dim=-1)



class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads      # 16
        self.n_kv = cfg.num_key_value_heads         # 8
        self.head_dim = cfg.head_dim                # 128
        # q_proj: cfg.hidden_size → n_heads * head_dim, bias=False
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        # k_proj: cfg.hidden_size → n_kv * head_dim,    bias=False
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        # v_proj: cfg.hidden_size → n_kv * head_dim,    bias=False
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        # o_proj: n_heads * head_dim → cfg.hidden_size, bias=False
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        # q_norm: RMSNorm(head_dim, eps=cfg.rms_norm_eps)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        # k_norm: RMSNorm(head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def qkv(self, x, cos, sin):
        B, S, _ = x.shape
        # 1. project      → q (B,S,2048), k and v (B,S,1024)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 2. view          → q (B,S,16,128), k and v (B,S,8,128)
        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_kv, self.head_dim)
        v = v.view(B, S, self.n_kv, self.head_dim)
        # 3. QK-norm       → q_norm on q, k_norm on k, nothing on v
        q = self.q_norm(q)
        k = self.k_norm(k)
        # 4. transpose(1,2)→ (B, heads, S, 128) for all three
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 5. apply_rope    → on q and k, nothing on v
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return q, k, v

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q, k, v = self.qkv(x, cos, sin)       # q (B,16,S,128), k and v (B,8,S,128)

        # 6. GQA — duplicate each K/V head in place: 8 → 16, order 0,0,1,1,...
        group = self.n_heads // self.n_kv 
        k = torch.repeat_interleave(k, group, dim=1)
        v = torch.repeat_interleave(v, group, dim=1)

        # 7. scores — every query against every key, per head → (B,16,S,S)
        # q @ k.transpose(-2, -1), then divide by math.sqrt(self.head_dim)
        scale = self.head_dim ** -0.5          # 1 / √128
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 8. mask — block looking ahead
        #    mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        #    scores = scores + mask
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask

        # 9. softmax over the last axis, in float32, then back to q's dtype
        #    torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        scores = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # 10. blend — weights @ v → (B,16,S,128)
        #     torch.matmul(scores, v)
        x = torch.matmul(scores, v)

        # 11. merge heads — (B,16,S,128) → (B,S,16,128) → (B,S,2048)
        #     .transpose(1, 2) then .reshape(B, S, self.n_heads * self.head_dim)
        x = x.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)

        # 12. output projection → (B,S,1024)
        #     return self.o_proj(...)
        return self.o_proj(x)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # gate_proj: cfg.hidden_size → cfg.intermediate_size, bias=False
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        # up_proj:   cfg.hidden_size → cfg.intermediate_size, bias=False
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        # down_proj: cfg.intermediate_size → cfg.hidden_size, bias=False
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        # gate = gate_proj(x)          (B,S,3072)
        gate = self.gate_proj(x)
        # up   = up_proj(x)(B,S,3072)
        up = self.up_proj(x)
        # h= F.silu(gate) * up (B,S,3072)  element by element
        h = F.silu(gate) * up
        # return down_proj(h)  (B,S,1024)
        return self.down_proj(h)

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # input_layernorm:          RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # self_attn:                Attention(cfg)
        self.self_attn = Attention(cfg)
        # post_attention_layernorm: RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # mlp:                      MLP(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        # x = x + attention of (normalized copy of x)
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        # x = x + mlp of (normalized copy of x)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x