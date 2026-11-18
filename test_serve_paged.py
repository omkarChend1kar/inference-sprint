import time
import torch
from transformers import AutoTokenizer
from model import load, generate_cached
from scheduler import Request, serve_paged
from paged import BlockPool

PATH = "../models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(PATH)
model = load(PATH)
eos = tok.eos_token_id

attn = model.layers[0].self_attn
pool = BlockPool(num_layers=len(model.layers), num_blocks=6, block_size=16,
                 n_kv=attn.n_kv, head_dim=attn.head_dim)
pool_bytes = sum(t.numel() * t.element_size() for t in pool.K + pool.V)
print(f"pool: 6 blocks x 16 = 96 tokens, {pool_bytes / 1e6:.1f} MB, allocated once\n")

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
reqs = [Request(rid, tok(p).input_ids, m, a) for rid, p, m, a in specs]

t0 = time.perf_counter()
done = serve_paged(model, reqs, pool, max_batch=8, eos_id=eos)
t_paged = time.perf_counter() - t0

print()
all_ok = True
for r in sorted(done, key=lambda r: r.rid):
    ref = generate_cached(model, torch.tensor([r.prompt]), r.max_new, eos)[0, len(r.prompt):].tolist()
    ok = ref == r.out
    all_ok &= ok
    print(f"{r.rid}  identical={ok}  {tok.decode(r.prompt + r.out)!r}")

n_tokens = sum(len(r.out) for r in done)
print(f"\nall identical:     {all_ok}")
print(f"blocks free at end: {len(pool.free)} / 6")
print(f"paged: {t_paged:.2f}s  {n_tokens / t_paged:.1f} tok/s")