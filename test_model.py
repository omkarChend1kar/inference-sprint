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