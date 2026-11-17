Thu Nov 12 14:02:24 UTC 2026: setup complete, oracle confirmed working on A5000, EU-SE-1
## Day 1 — done Mon 16 Nov
- DecoderLayer bit-exact vs HF Qwen3 layer 0 (all sub-tests 0.0)
- KV cache per token: 2 (K and V) × 28 layers × 8 KV heads × 128 head_dim × 2 bytes = 114,688 bytes per token
- At that rate, 1 GB of KV cache holds about : 1,073,741,824 ÷ 114,688 ≈ 9,362 tokens
- One gigabyte of KV cache holds about 9,400 tokens — across all users combined. A 24 GB A5000, after the 1.2 GB of weights, has room for roughly 200,000 tokens of cache. If each user’s conversation is 2,000 tokens, that’s about 100 users at once.

## Day 2 — Mon 16 Nov
- Full forward pass from safetensors: logits bit-exact vs HF (diff 0.0)
- Greedy generate, 30 tokens: identical to HF
- Baseline, no cache: 9.46 s / 30 tokens (~3.2 tok/s)
  Intel Mac CPU, float32, 5-token prompt

## Day 3 — Mon 16 Nov
- KV cache: generated tokens identical to no-cache version
- 30 tokens: 2.91 s with cache vs 32.83 s without → 11.3x
  (baseline varied 9.46 s → 32.83 s between runs: single timings unreliable)
- Speedup < 17x work ratio: each decode pass reads all weights regardless
  of token count → decode is memory-bandwidth-bound

## Day 3 — prefill vs decode (median of 3, after warm-up, own model only)
| prompt | TTFT s | TPOT s |
|   5    | 0.167  | 0.117  |
|  50    | 0.992  | 0.140  |
| 200    | 2.460  | 0.157  |
| 500    | 4.126  | 0.165  |
- TTFT grows sublinearly: per-token prefill cost falls 33 → 8 ms
- TPOT nearly flat: a decode pass ≈ one read of the weights
- 2.4 GB / 0.117 s ≈ 20 GB/s → decode runs at RAM bandwidth (memory-bound)
- Extra tokens in a pass are cheap (~12 ms) vs the pass itself (~117 ms) → batching

## Day 4 — Tue 17 Nov
- Continuous batching: 4 requests, staggered arrivals, max_batch=3
- Every request's output identical to running alone
- Batched 17.0 tok/s vs sequential 9.6 tok/s → 1.8x
  (batch rarely full; per-user attention loop in Python; batched run went first, unwarmed)
- Batching raises throughput, slightly raises each request's per-token latency
- Trace matched prediction: max_batch=3 → 22 passes, avg batch 2.6, D queued 7 passes
- max_batch=4 → 20 passes, avg batch 2.9, D admitted on arrival
- Run-to-run noise ±10–15%: single runs can't rank 3 vs 4