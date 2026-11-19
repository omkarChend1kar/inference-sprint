import time
import torch
from transformers import AutoTokenizer
from model import load, generate_cached
from scheduler import Request, serve_preempt
from paged import BlockPool

PATH = "../models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(PATH)
model = load(PATH)
eos = tok.eos_token_id
attn = model.layers[0].self_attn

specs = [
    ("A", "The capital of France is",                    20, 0),
    ("B", "Water boils at a temperature of",             12, 0),
    ("C", "The largest planet in the solar",             16, 1),
    ("D", "Python is a programming language",            10, 2),
    ("E", "The speed of light is",                       24, 3),
    ("F", "Mount Everest is located in",                  8, 4),
    ("G", "The chemical symbol for gold is",             14, 6),
    ("H", "Shakespeare wrote the play",                  18, 8),
    ("I", "The first person to walk on the moon was",    22, 10),
    ("J", "Photosynthesis is the process by which",       9, 12),
]

# reference outputs, computed once
refs = {}
for rid, p, m, _ in specs:
    ids = tok(p).input_ids
    refs[rid] = generate_cached(model, torch.tensor([ids]), m, eos)[0, len(ids):].tolist()

for num_blocks in [6, 3]:
    print(f"\n========== pool of {num_blocks} blocks ==========")
    pool = BlockPool(num_layers=len(model.layers), num_blocks=num_blocks, block_size=16,
                     n_kv=attn.n_kv, head_dim=attn.head_dim)
    reqs = [Request(rid, tok(p).input_ids, m, a) for rid, p, m, a in specs]

    t0 = time.perf_counter()
    done = serve_preempt(model, reqs, pool, max_batch=8, eos_id=eos)
    t = time.perf_counter() - t0

    all_ok = all(refs[r.rid] == r.out for r in done)
    n_tokens = sum(len(r.out) for r in done)
    print(f"all identical: {all_ok}")
    print(f"blocks free at end: {len(pool.free)} / {num_blocks}")
    print(f"{t:.2f}s  {n_tokens / t:.1f} tok/s")