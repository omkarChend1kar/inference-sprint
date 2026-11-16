import time, statistics
import torch
from transformers import AutoTokenizer
from model import load
from layer import LayerCache

PATH = "../models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(PATH)
model = load(PATH)

@torch.no_grad()
def measure(model, ids, n_decode=20):
    caches = [LayerCache() for _ in model.layers]
    # PREFILL — time it
    t0 = time.perf_counter()
    # run the whole prompt with start=0, take argmax of last row → next_id
    logits = model(ids, caches, start_index=0)
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    ttft = time.perf_counter() - t0

    # DECODE — time n_decode passes together
    pos = ids.shape[1]
    t1 = time.perf_counter()
    # repeat n_decode times: one-token pass at start=pos, pos += 1, new next_id
    for _ in range(n_decode):
        logits = model(ids[:, -1:], caches, start_index=pos)
        pos += 1
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    tpot = (time.perf_counter() - t1) / n_decode

    return ttft, tpot

text = "The capital of France is Paris. " * 200
all_ids = tok(text, return_tensors="pt").input_ids

measure(model, all_ids[:, :5], 3)          # warm-up run, result discarded

print(f"{'prompt':>6} {'TTFT s':>8} {'TPOT s':>8}")
for L in [5, 50, 200, 500]:
    ids = all_ids[:, :L]
    runs = [measure(model, ids) for _ in range(3)]
    ttft = statistics.median(r[0] for r in runs)
    tpot = statistics.median(r[1] for r in runs)
    print(f"{L:>6} {ttft:>8.3f} {tpot:>8.3f}")