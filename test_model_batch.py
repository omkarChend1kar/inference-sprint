from layer import LayerCache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import load

PATH = "../models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(PATH)
hf = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32,
                                          attn_implementation="eager").eval()
mine = load(PATH)

# two prompts of different lengths
pA = tok("The capital of France is", return_tensors="pt").input_ids[0]
pB = tok("Water boils at a temperature of", return_tensors="pt").input_ids[0]
nA, nB = len(pA), len(pB)

# one list of 28 empty caches per user
caches = [[LayerCache() for _ in mine.layers] for _ in range(2)]

with torch.no_grad():
    # both prompts prefilled together in ONE pass
    logits = mine.forward_batch(
        torch.cat([pA, pB]),                              # ids, flattened pile
        torch.cat([torch.arange(nA), torch.arange(nB)]),  # each user's own positions
        [(0, nA), (nA, nB)],                              # who owns which rows
        caches,
    )
    # reference: each prompt alone, last row only
    aloneA = mine(pA[None])[0, -1]
    aloneB = mine(pB[None])[0, -1]

print("batched logits", tuple(logits.shape))
print("model batch A", (logits[0] - aloneA).abs().max().item())
print("model batch B", (logits[1] - aloneB).abs().max().item())
print("A next:", repr(tok.decode(logits[0].argmax())))
print("B next:", repr(tok.decode(logits[1].argmax())))