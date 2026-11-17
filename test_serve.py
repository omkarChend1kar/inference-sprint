import time
import torch
from transformers import AutoTokenizer
from model import load, generate_cached
from scheduler import Request, serve

PATH = "../models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(PATH)
model = load(PATH)
eos = tok.eos_token_id

prompts = [
    ("A", "The capital of France is",          20, 0),
    ("B", "Water boils at a temperature of",   12, 0),
    ("C", "The largest planet in the solar",   16, 2),
    ("D", "Python is a programming language", 10, 5),
]
reqs = [Request(rid, tok(p).input_ids, m, arr) for rid, p, m, arr in prompts]

t0 = time.perf_counter()
done = serve(model, reqs, max_batch=3, eos_id=eos)
t_batch = time.perf_counter() - t0

# reference: each request alone, one after another
t0 = time.perf_counter()
all_ok = True
for r in sorted(done, key=lambda r: r.rid):
    ids = torch.tensor([r.prompt])
    ref = generate_cached(model, ids, r.max_new, eos)[0, len(r.prompt):].tolist()
    ok = ref == r.out
    all_ok &= ok
    print(f"{r.rid}  identical={ok}  {tok.decode(r.prompt + r.out)!r}")
t_seq = time.perf_counter() - t0

n_tokens = sum(len(r.out) for r in done)
print(f"\nall identical: {all_ok}")
print(f"batched:    {t_batch:.2f}s  {n_tokens / t_batch:.1f} tok/s")
print(f"sequential: {t_seq:.2f}s  {n_tokens / t_seq:.1f} tok/s")