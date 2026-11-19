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

## Day 5 — Wed 18 Nov — GATE PASSED
- Paged KV cache: BlockPool (fixed tensors per layer + free list), PagedSeq (block table),
  PagedLayer (same append() interface — attention unchanged)
- 10 requests, staggered, 6-block pool (96 tokens, 22 MB, allocated once):
  all identical to running alone; 6/6 blocks free at end
- max_batch=8 but pile never > 4: admission limited by memory
- Conservative admission (reserve worst case) left 3/6 blocks idle while 6 requests
  queued (passes 2–10) → motivates optimistic admission + preemption (vLLM)

## Thu 19 Nov — optimistic admission + preemption
- Admit if current feed fits (after this pass's growth); preempt newest when growth > free
- Preempted request readmitted with prompt + generated tokens; outputs still identical
- 6 blocks: D admitted pass 2 (vs 12); 42 passes (vs 56), avg batch 3.6 (vs 2.7),
  3 preemptions, 33 tokens recomputed
- 3 blocks: 6 preemptions, 88 tokens recomputed; C preempted twice in 3 passes (thrashing)
- Headroom fix: admission reserves next-token room for ALL running requests (growth_next)
  3-block pool: preemptions 6 → 5, recomputed tokens 88 → 77 → 71
  Remaining preemptions = lost optimistic bets; fewer requires a larger reserve (idle memory)
- Same 6-block schedule timed 3×: 10.40 / 8.11 / 7.45 s → ±40%, single runs meaningless

## Day 6 setup — Thu 19 Nov
- GPU: RTX A6000 48 GB, RunPod EU-SE-1, $0.70/hr
- vLLM 0.34.0, own venv at /workspace/vllm-env (~20 GB — CUDA 12 + 13 libs)
- Install into network volume: ~25 min; use tmux for long installs
