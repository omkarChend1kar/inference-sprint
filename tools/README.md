# llm-ceiling

A one-file estimator for how fast a dense LLM can decode on one NVIDIA GPU, and where its output throughput tops out. It reads the model's `config.json`, asks `nvidia-smi` which GPU you're on, and prints each step of the math so you can check it.

```bash
./llm-ceiling.sh /path/to/model                      # on the GPU machine
./llm-ceiling.sh /path/to/model --in 1024 --out 128  # different request shape
./llm-ceiling.sh /path/to/model --gpu H100 --mem-gib 79.6   # a GPU you don't have
./llm-ceiling.sh /path/to/model --calibrate 3.40     # your measured 1-user TPOT, in ms
```

The only requirement is `python3`.

## The derivation

Decode produces one token per request per **pass**, which is one run through all the layers. Each pass has two jobs: carry bytes from GPU memory to the compute units, and do the math on them. At normal batch sizes the carrying takes far longer, so it sets the time.

| step | quantity | formula | Qwen3-0.6B, 128 in / 128 out |
|---|---|---|---|
| 1 | weight bytes per pass | params × bytes per weight | 0.596 B × 2 = **1.19 GB** |
| 2 | KV bytes per token | 2 (K and V) × layers × KV heads × head_dim × bytes | 2 × 28 × 8 × 128 × 2 = **114,688 B** |
| 3 | KV bytes per user | (input + output/2) tokens × step 2 | 192 × 114,688 = **22.0 MB** |
| 4 | pass time = TPOT | (weights + users × KV per user) ÷ achieved bandwidth | (1.19 GB + c × 22 MB) ÷ 348 GB/s |
| 5 | output tokens/s | users ÷ TPOT | |
| 6 | memory cap | (GPU memory × 0.9 − weights) ÷ step 2 ÷ (input + output) | 1,440 requests on an A40 |
| 7 | ceiling | achieved bandwidth ÷ KV per user | 348 GB/s ÷ 22 MB ≈ **15,800 tok/s** |

Why it has this shape:

- **The weights are shared.** In a batch, all requests' tokens go through each matrix together, so the weights are carried once per pass however many users there are. That's why going from 1 to 64 users costs only about 2× in TPOT but gives 64× the tokens per pass.
- **The KV cache is not shared.** Each request reads its own K and V, so this cost grows with every user. Once it outweighs the weights, each extra token costs its own KV read, and throughput flattens toward step 7.
- **The request shape matters.** Longer prompts mean more KV per user. Here 1,024-token prompts drop the A40 ceiling from about 15,800 to about 2,800 tok/s.
- **GQA helps directly.** Fewer KV heads means fewer KV bytes per token, which raises the ceiling by the same factor.
- **Open-loop saturation:** max output tok/s ÷ output tokens per request = the highest request rate the server can keep up with. Send more than that and the waiting line grows without limit.

The script also checks the math limit (2 × params FLOPs per token, plus attention), and reports which one binds. Small models stay memory-bound. Larger ones can become math-bound at high batch sizes, as Llama-3.1-8B on an H100 does from about 500 users.

## Validation

vLLM 0.34, Qwen3-0.6B in bfloat16, one A40, 128 in / 128 out, closed loop (`--max-concurrency c`), default `--eff 0.5`:

| users | predicted TPOT | measured median TPOT | error |
|---|---|---|---|
| 1 | 3.49 ms | 3.40 ms | +2.6% |
| 4 | 3.68 ms | 3.70 ms | −0.5% |
| 16 | 4.44 ms | 4.52 ms | −1.8% |
| 64 | 7.48 ms | 7.91 ms | −5.4% |

## Achieved fraction (`--eff`)

No GPU reaches its spec-sheet bandwidth. The default of 0.5 is what vLLM achieved for this 0.6B model on an A40 (1.2 GB in 3.40 ms ≈ 353 of 696 GB/s). Bigger models usually get a higher fraction, because each step moves more data and fixed overheads matter less. The best value is your own: run one request at a time, take the median TPOT, and pass it as `--calibrate`.

## What it leaves out

- **Prefill.** Prompt processing takes passes too. Measured throughput at high load came out about 20% below this estimate in the A40 runs.
- **Server caps.** vLLM limits requests per pass (`--max-num-seqs`). Pass the same value as `--max-seqs`.
- **Overlap.** It takes the slower of carrying and math, as if they overlapped perfectly.
- **Not supported:** mixture-of-experts, DeepSeek-style MLA, sliding-window attention, multi-GPU (tensor parallelism), non-SwiGLU MLPs.
- **GPU table.** Peak numbers come from vendor spec sheets for common data-center and workstation cards. For anything else, pass `--bw` (and `--tflops`).

Treat the output as a first estimate to check against a real benchmark, not as a replacement for one.
