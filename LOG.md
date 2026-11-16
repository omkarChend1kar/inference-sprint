Thu Nov 12 14:02:24 UTC 2026: setup complete, oracle confirmed working on A5000, EU-SE-1
## Day 1 — done Mon 16 Nov
- DecoderLayer bit-exact vs HF Qwen3 layer 0 (all sub-tests 0.0)
- KV cache per token: 2 (K and V) × 28 layers × 8 KV heads × 128 head_dim × 2 bytes = 114,688 bytes per token
- At that rate, 1 GB of KV cache holds about : 1,073,741,824 ÷ 114,688 ≈ 9,362 tokens
- One gigabyte of KV cache holds about 9,400 tokens — across all users combined. A 24 GB A5000, after the 1.2 GB of weights, has room for roughly 200,000 tokens of cache. If each user’s conversation is 2,000 tokens, that’s about 100 users at once.