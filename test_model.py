import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import load

PATH = "../models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(PATH)
hf = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32,
                                          attn_implementation="eager").eval()
mine = load(PATH)

ids = tok("The capital of France is", return_tensors="pt").input_ids
print("ids", ids.tolist())

with torch.no_grad():
    a = mine(ids)
    b = hf(ids).logits
    print("logits shape", tuple(a.shape))
    print("logits diff ", (a - b).abs().max().item())
    print("mine next:", repr(tok.decode(a[0, -1].argmax())))
    print("hf   next:", repr(tok.decode(b[0, -1].argmax())))

import time
from model import generate

N = 30
eos = tok.eos_token_id

t0 = time.perf_counter()
mine_ids = generate(mine, ids, N, eos)
t_mine = time.perf_counter() - t0

with torch.no_grad():
    hf_ids = hf.generate(ids, max_new_tokens=N, do_sample=False)

n = min(mine_ids.shape[1], hf_ids.shape[1])
print("identical tokens:", torch.equal(mine_ids[:, :n], hf_ids[:, :n]))
print("mine:", repr(tok.decode(mine_ids[0])))
print("hf:  ", repr(tok.decode(hf_ids[0])))
print(f"time: {t_mine:.2f}s for {mine_ids.shape[1] - ids.shape[1]} new tokens, no cache")

from model import generate_cached

t0 = time.perf_counter()
cached_ids = generate_cached(mine, ids, N, eos)
t_cached = time.perf_counter() - t0

print("cache identical:", torch.equal(cached_ids, mine_ids))
print(f"time: {t_cached:.2f}s with cache  vs  {t_mine:.2f}s without")
print(f"speedup: {t_mine / t_cached:.1f}x")