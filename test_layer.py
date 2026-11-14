from layer import RMSNorm, apply_rope, rope_tables
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
from layer import RMSNorm, rope_tables
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL_PATH = "../models/Qwen3-0.6B"
torch.manual_seed(0)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
).to(device)
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



