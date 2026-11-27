# llm-ceiling

A one-file estimator for how fast a dense LLM can decode on one NVIDIA GPU, where its throughput tops out, and which batch cap keeps a latency promise. It reads the model's `config.json`, asks `nvidia-smi` which GPU you're on, and prints every step of the math so you can check it.

```bash
./llm-ceiling.sh /path/to/model --max-seqs 256        # match your server's cap (vLLM --max-num-seqs)
./llm-ceiling.sh /path/to/model --in 1024 --out 128   # a different request shape
./llm-ceiling.sh /path/to/model --gpu H100 --mem-gib 79.6   # a GPU you don't have
./llm-ceiling.sh /path/to/model --calibrate 3.40      # your measured 1-user TPOT, in ms
```

The only requirement is `python3`.

## The model

Decode produces one token per running request per **pass**. A GPU has two speeds that matter: how fast memory can deliver bytes (the **belt**, memory bandwidth) and how fast it can do arithmetic (the **workshop**, FLOP/s). A pass is two steps that happen one after the other:

1. **Linear layers** (q/k/v/o projections, MLP, LM head) multiply every token in the pass by the weights. The belt has to bring all the weights, and the workshop has to do 2 FLOPs per weight per token. Both happen at the same time, so this step takes **whichever is slower**.
2. **Attention** reads each running request's own K and V. That's belt work, and it grows with every user.

In a busy server, requests keep finishing and new ones join. Each newcomer's whole prompt goes through the linear layers in the same pass as everyone's single decode token.

| step | quantity | formula | Qwen3-0.6B on an A40, 128 in / 128 out |
|---|---|---|---|
| 1 | weights per pass | params × bytes per weight | 1.19 GB → 3.43 ms on the belt |
| 2 | KV bytes per token | 2 × layers × KV heads × head_dim × bytes | 114,688 B |
| 3 | KV bytes per user | (input + output/2) × step 2 | 22.0 MB |
| 4 | tokens per pass | users × (1 + input/output) | 512 at 256 users |
| 5 | pass time = TPOT | max(weights ÷ belt, tokens × 2·params ÷ workshop) + users × KV per user ÷ belt | 12.2 + 16.2 = 28.4 ms at 256 users |
| 6 | memory cap | (GPU memory × 0.9 − weights) ÷ step 2 ÷ (input + output) | 1,440 requests |
| 7 | ceiling | 1 ÷ (math per output token + KV carrying per output token) | 47.7 µs + 63.3 µs → ~9,000 tok/s |

What follows from it:

- **Small batches are nearly free.** The weights are carried once and shared, so going from 1 to 64 users costs about 2× in TPOT and gives about 30× the tokens per second.
- **Past a point, every user pays full price.** Once the linear step is workshop-bound and KV dominates the belt, both steps grow in proportion to the number of users. TPOT then grows linearly and throughput stops rising. On the A40 this happens from about 128 users.
- **So the batch cap is a latency knob, not a throughput knob, once you're past that point.** vLLM's default of 256 gave 28 ms TPOT. About 180 users gives the same throughput at ≤ 20 ms.
- **The request shape matters.** 1,024-token prompts cut the A40 ceiling from about 9,000 to about 1,750 tok/s: more KV per user, and more prompt math per output token.
- **Open-loop knee** = max tok/s ÷ output tokens per request. Send more requests per second than that and the waiting line grows without limit.

## Validation

vLLM 0.34, Qwen3-0.6B in bfloat16, one A40, 128 in / 128 out, `--max-seqs 256`. Two numbers were **fitted** to measurements; everything else is a **prediction**.

| | predicted | measured | |
|---|---|---|---|
| TPOT, 1 user (closed loop) | 3.49 ms | 3.40 ms | fitted (sets belt efficiency 0.5) |
| TPOT, 4 users | 3.68 ms | 3.70 ms | −0.5% |
| TPOT, 16 users | 4.44 ms | 4.52 ms | −1.8% |
| TPOT, 64 users | 7.48 ms | 7.91 ms | −5.4% |
| TPOT, 256 running (saturated, open loop) | 28.4 ms | 28.2 ms | fitted (sets workshop efficiency 0.33) |
| open-loop knee | ~70 req/s | ~68 req/s | +3% |

The 256-request cap, the knee and the growth of the waiting line were confirmed with a live scheduler log during a rate-96 run: `running` held at 256, `waiting` grew about 27/s (predicted 96 − 70 = 26/s), and the GPU stayed 100% busy.

## Efficiencies (`--eff`, `--math-eff`)

No GPU reaches its spec sheet. The defaults come from the A40 runs above:

- **belt 0.5**: 1.19 GB in 3.40 ms at 1 user ≈ 353 of 696 GB/s. Set your own with `--calibrate <1-user TPOT in ms>`.
- **workshop 0.33**: the value that makes a saturated pass take 28 ms. This 0.6B model has small matrices, which keep the GPU's math units from running at full speed. Bigger models usually reach a higher fraction, so for them the default is pessimistic. Measure a saturated pass and adjust.

## What it leaves out

- **Time before the first token.** HTTP, tokenizing and scheduling added 25–150 ms per request in the A40 runs. That lowers measured throughput in closed-loop tests (by about 20% at 64 users).
- **Random arrivals.** Real traffic bunches up. Near the knee (rate 64 of a ~70 max), that raised measured TPOT to 13 ms against 7–9 ms predicted.
- **Not supported:** mixture-of-experts, DeepSeek-style MLA, sliding-window attention, multi-GPU, non-SwiGLU MLPs. Peak numbers come from vendor spec sheets for common cards; for anything else, pass `--bw` and `--tflops`.

Treat the output as a first estimate to check against a real benchmark, not as a replacement for one.
