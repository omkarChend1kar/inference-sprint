from layer import DecoderLayer, RMSNorm, apply_rope, rope_tables, MLP, Attention
import torch
n = RMSNorm(2)
# print(n(torch.tensor([3.0, 4.0])))


cos, sin = rope_tables(128, 32)
# print(cos.shape)      # torch.Size([32, 128])
# print(cos[0][:5])     # all 1.0 — cos(0) = 1
# print(sin[0][:5])     # all 0.0 — sin(0) = 0
# print(cos[1][0], cos[1][64])   # should be equal — the pairing


cos, sin = rope_tables(4, 3, theta=100.0)
x3 = torch.tensor([[1.0,0,0,1],[1.0,0,0,1]]).view(1,1,2,4)
# print(apply_rope(x3, cos, sin))

import torch
from transformers import AutoModelForCausalLM
from layer import RMSNorm, rope_tables, Attention, apply_rope, MLP
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL_PATH = "../models/Qwen3-0.6B"
torch.manual_seed(0)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, attn_implementation="eager"
)
model.eval()
hf_layer = model.model.layers[0]

def diff(a, b):
    return (a - b).abs().max().item()

x = torch.randn(1, 5, 1024)

with torch.no_grad():
    # 1. RMSNorm with layer 0's real trained weight
    mine = RMSNorm(1024)
    mine.load_state_dict(hf_layer.input_layernorm.state_dict())
    print("rmsnorm      ", diff(mine(x), hf_layer.input_layernorm(x)))

    # 2. RoPE tables against HF's own
    cos, sin = rope_tables(128, 32)
    pos_ids = torch.arange(5)[None]
    hf_cos, hf_sin = model.model.rotary_emb(x, pos_ids)
    print("rope cos     ", diff(cos[:5], hf_cos[0]))
    print("rope sin     ", diff(sin[:5], hf_sin[0]))
    # 3. RoPE rotation itself
    q = torch.randn(1, 16, 5, 128)   # batch, Q heads, tokens, head_dim
    k = torch.randn(1, 8, 5, 128)    # K has only 8 heads
    hf_q, hf_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    print("rope rotate q", diff(apply_rope(q, cos, sin), hf_q))
    print("rope rotate k", diff(apply_rope(k, cos, sin), hf_k))

    # 5. Attention part 1 — Q, K, V
    hf_attn = hf_layer.self_attn
    mine_attn = Attention(model.config)
    mine_attn.load_state_dict(hf_attn.state_dict())
    q, k, v = mine_attn.qkv(x, cos, sin)

    # reference: the same five steps using HF's own pieces
    B, S, _ = x.shape
    hq = hf_attn.q_norm(hf_attn.q_proj(x).view(B, S, -1, 128)).transpose(1, 2)
    hk = hf_attn.k_norm(hf_attn.k_proj(x).view(B, S, -1, 128)).transpose(1, 2)
    hv = hf_attn.v_proj(x).view(B, S, -1, 128).transpose(1, 2)
    hq, hk = apply_rotary_pos_emb(hq, hk, hf_cos, hf_sin)

    print("shapes       ", tuple(q.shape), tuple(k.shape), tuple(v.shape))
    print("attn q       ", diff(q, hq))
    print("attn k       ", diff(k, hk))
    print("attn v       ", diff(v, hv))

    # 6. Attention full — the whole module, x in, update out
    mask = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)[None, None]
    hf_out, _ = hf_attn(x, position_embeddings=(hf_cos, hf_sin), attention_mask=mask)
    mine_out = mine_attn(x, cos, sin)
    print("attn shape   ", tuple(mine_out.shape))
    print("attn full    ", diff(mine_out, hf_out))

    # 7. MLP
    mine_mlp = MLP(model.config)
    mine_mlp.load_state_dict(hf_layer.mlp.state_dict())
    print("mlp     ", diff(mine_mlp(x), hf_layer.mlp(x)))


    # 8. Full decoder layer — the Day 1 gate
    mine_layer = DecoderLayer(model.config)
    mine_layer.load_state_dict(hf_layer.state_dict())
    hf_layer_out = hf_layer(x, attention_mask=mask, position_embeddings=(hf_cos, hf_sin))[0]
    print("layer        ", diff(mine_layer(x, cos, sin), hf_layer_out))
