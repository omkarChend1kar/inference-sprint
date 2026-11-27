#!/usr/bin/env bash
# llm-ceiling.sh (v2): back-of-envelope decode speed and throughput ceiling for a
# dense transformer LLM on one NVIDIA GPU, printed step by step.
#
#   ./llm-ceiling.sh /path/to/model --max-seqs 256            # match your server's cap
#   ./llm-ceiling.sh /path/to/model --in 1024 --out 128
#   ./llm-ceiling.sh /path/to/model --gpu H100 --mem-gib 79.6  # a GPU you don't have
#   ./llm-ceiling.sh /path/to/model --calibrate 3.40           # your measured 1-user TPOT (ms)
#
# One decode pass = two steps, one after the other:
#   linear layers: the slower of (carry all weights) and (math for every token in the pass)
#   attention:     carry every running request's own KV cache
# Tokens in a pass = 1 per running request + the prompts of requests joining this pass.
# See README.md for the derivation and its limits.

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "llm-ceiling: needs python3" >&2
  exit 1
fi

GPU_LINE=""
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits -i 0 2>/dev/null || true)
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || echo 0)
fi

exec python3 - "$GPU_LINE" "$GPU_COUNT" "$@" <<'PY'
import argparse, json, math, os, re, sys

gpu_line, gpu_count = sys.argv[1], int(sys.argv[2] or 0)

# nvidia-smi name pattern, label, peak memory bandwidth (GB/s),
# peak dense BF16 tensor math (TFLOP/s; None = no BF16 or not listed).
# Order matters: more specific names first.
GPU_TABLE = [
    (r"B200",             "B200",           8000, 2250),
    (r"GH200",            "GH200",          None, None),
    (r"H200",             "H200",           4800,  989),
    (r"H100.*NVL",        "H100 NVL",       3900, None),
    (r"H100.*PCIe",       "H100 PCIe",      2000,  756),
    (r"H100",             "H100 SXM",       3350,  989),
    (r"A100.*80GB.*PCIe", "A100 80GB PCIe", 1935,  312),
    (r"A100.*80GB",       "A100 80GB SXM",  2039,  312),
    (r"A100",             "A100 40GB",      1555,  312),
    (r"L40S",             "L40S",            864,  362),
    (r"\bL40\b",          "L40",             864,  181),
    (r"\bL4\b",           "L4",              300,  121),
    (r"RTX 6000 Ada",     "RTX 6000 Ada",    960,  364),
    (r"RTX A6000",        "RTX A6000",       768,  155),
    (r"\bA40\b",          "A40",             696,  150),
    (r"A10G",             "A10G",            600, None),
    (r"\bA10\b",          "A10",             600,  125),
    (r"4090",             "RTX 4090",       1008,  165),
    (r"3090",             "RTX 3090",        936,   71),
    (r"\bT4\b",           "T4",              320, None),
    (r"V100",             "V100",            900, None),
]

def die(msg):
    print(f"llm-ceiling: {msg}", file=sys.stderr)
    sys.exit(2)

p = argparse.ArgumentParser(
    prog="llm-ceiling.sh",
    description="Estimate decode TPOT and the output-throughput ceiling of a dense LLM on one GPU.")
p.add_argument("model", help="model folder containing config.json (or the config.json itself)")
p.add_argument("--in", dest="inp", type=int, default=128, help="prompt tokens per request (default 128)")
p.add_argument("--out", type=int, default=128, help="output tokens per request (default 128)")
p.add_argument("--gpu", help="GPU name to look up instead of asking nvidia-smi, e.g. A40, H100, 'A100 80GB'")
p.add_argument("--mem-gib", type=float, help="GPU memory in GiB (default: from nvidia-smi)")
p.add_argument("--bw", type=float, help="peak memory bandwidth in GB/s (default: from the built-in table)")
p.add_argument("--tflops", type=float, help="peak dense BF16 TFLOP/s (default: from the built-in table)")
p.add_argument("--eff", type=float, default=0.5,
               help="fraction of peak bandwidth achieved (default 0.5: vLLM, Qwen3-0.6B, A40 at 1 user)")
p.add_argument("--math-eff", type=float, default=1/3,
               help="fraction of peak math achieved (default 0.33: vLLM, Qwen3-0.6B, A40 at saturation)")
p.add_argument("--calibrate", type=float, metavar="TPOT_MS",
               help="your measured median TPOT at 1 user, in ms; sets --eff from it")
p.add_argument("--weight-bytes", type=float, default=2, help="bytes per weight: 2 bf16/fp16, 1 fp8/int8, 0.5 int4")
p.add_argument("--kv-bytes", type=float, default=2, help="bytes per KV-cache value: 2 bf16, 1 fp8")
p.add_argument("--gpu-util", type=float, default=0.9,
               help="share of GPU memory the server may use (vLLM --gpu-memory-utilization, default 0.9)")
p.add_argument("--max-seqs", type=int, help="server's cap on requests per pass (vLLM --max-num-seqs)")
p.add_argument("--tpot-max", type=float, default=20, help="per-token latency promise in ms (default 20)")
a = p.parse_args(sys.argv[3:])

# ---------- model ----------
cfg_path = os.path.join(a.model, "config.json") if os.path.isdir(a.model) else a.model
if not os.path.isfile(cfg_path):
    die(f"no config.json at {cfg_path}")
with open(cfg_path) as f:
    cfg = json.load(f)
c = cfg.get("text_config", cfg)
for k in ("num_experts", "num_local_experts", "n_routed_experts"):
    if (c.get(k) or 0) > 1:
        die("mixture-of-experts models are not supported (each pass reads only some experts)")
if c.get("kv_lora_rank"):
    die("MLA attention (DeepSeek-style compressed KV) is not supported")
try:
    H, L, nh = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
    inter, vocab = c["intermediate_size"], c["vocab_size"]
except KeyError as e:
    die(f"config.json is missing {e}; is this a decoder-only transformer?")
nkv = c.get("num_key_value_heads") or nh
hd = c.get("head_dim") or H // nh
tied = bool(c.get("tie_word_embeddings", False))
model_name = os.path.basename(os.path.normpath(a.model if os.path.isdir(a.model) else os.path.dirname(cfg_path)))

attn_w = H * nh * hd + 2 * H * nkv * hd + nh * hd * H  # q, k, v, o projections
mlp_w = 3 * H * inter                                  # gate, up, down (SwiGLU)
per_layer = attn_w + mlp_w + 2 * H                     # + two norm weight vectors
params = L * per_layer + vocab * H + H + (0 if tied else vocab * H)
w_stored = params * a.weight_bytes
# An untied input embedding is a table lookup: only the current tokens' rows are read,
# and it does no matrix math. Everything else is carried and multiplied every pass.
matmul_params = params - (0 if tied else vocab * H)
w_read = matmul_params * a.weight_bytes
flops_tok = 2 * matmul_params                          # one multiply + one add per weight
kv_tok = 2 * L * nkv * hd * a.kv_bytes                 # K and V, every layer, every KV head
avg_ctx = a.inp + a.out / 2                            # tokens a request holds, averaged over its decode
peak_ctx = a.inp + a.out                               # tokens it holds at the end
kv_user = avg_ctx * kv_tok
prompt_share = a.inp / a.out                           # prompt tokens riding along per decode token

# ---------- GPU ----------
name = a.gpu or (gpu_line.split(",")[0].strip() if gpu_line else None)
if not name:
    die("no NVIDIA GPU found. Pass --gpu NAME (and --mem-gib) to estimate for a GPU you don't have.")
label, t_bw, t_tf = None, None, None
for rx, lab, bw, tf in GPU_TABLE:
    if re.search(rx, name, re.I):
        label, t_bw, t_tf = lab, bw, tf
        break
peak_bw = a.bw or t_bw
if not peak_bw:
    die(f"don't know the memory bandwidth of '{name}'. Look it up on the spec sheet and pass --bw GB/s.")
peak_tf = a.tflops or t_tf
mem_gib = a.mem_gib or (float(gpu_line.split(",")[1]) / 1024 if (gpu_line and not a.gpu) else None)

eff = a.eff
if a.calibrate:
    # at 1 user the pass is: carry weights + carry one user's KV (math is tiny)
    eff = (w_read + kv_user) / (a.calibrate / 1000 * peak_bw * 1e9)
    if eff > 1:
        die(f"--calibrate {a.calibrate} ms would mean {eff:.0%} of peak bandwidth; check the number or --bw")
B = peak_bw * 1e9 * eff
F = peak_tf * 1e12 * a.math_eff if peak_tf else None

def tokens_in_pass(users):
    # steady state: users/out requests finish per pass, as many join, each bringing its prompt
    return users * (1 + prompt_share)

def pass_parts(users):
    carry_w = w_read / B
    math = tokens_in_pass(users) * flops_tok / F if F else 0.0
    linear, who = (math, "workshop") if math > carry_w else (carry_w, "belt")
    attention = users * kv_user / B
    return linear, who, attention

def pass_time(users):
    linear, _, attention = pass_parts(users)
    return linear + attention

# ---------- capacity ----------
c_mem = None
if mem_gib:
    kv_space = mem_gib * 2**30 * a.gpu_util - w_stored
    if kv_space <= 0:
        die(f"weights ({w_stored/1e9:.1f} GB) don't fit in {a.gpu_util:.0%} of {mem_gib:.1f} GiB")
    max_tokens = kv_space / kv_tok
    c_mem = int(max_tokens // peak_ctx)
caps = [(v, why) for v, why in [(c_mem, "GPU memory"), (a.max_seqs, "--max-seqs")] if v]
c_cap, cap_why = min(caps) if caps else (None, None)

# cost of one output token once there are many users: its share of the math + its user's KV
cost_math = (1 + prompt_share) * flops_tok / F if F else 0.0
cost_kv = kv_user / B
tok_inf = 1 / (cost_math + cost_kv)

def max_users_under(limit_s):
    hi = c_cap or 10**6
    if pass_time(1) > limit_s:
        return 0
    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pass_time(mid) <= limit_s:
            lo = mid
        else:
            hi = mid - 1
    return lo

# ---------- print ----------
def gb(x):  return f"{x/1e9:,.2f} GB"
def mb(x):  return f"{x/1e6:,.1f} MB"
def n(x):   return f"{x:,.0f}"

print()
print(f"GPU       {name}" + (f"  (table entry: {label})" if label else "") +
      (f"  x{gpu_count} found, using one" if gpu_count > 1 else ""))
if mem_gib:
    print(f"          memory {mem_gib:,.1f} GiB")
print(f"          belt (memory bandwidth)  {n(peak_bw)} GB/s peak x {eff:.2f} = {n(B/1e9)} GB/s"
      + ("   (from --calibrate)" if a.calibrate else ""))
if F:
    print(f"          workshop (BF16 math)     {n(peak_tf)} TFLOP/s peak x {a.math_eff:.2f} = {n(F/1e12)} TFLOP/s")
else:
    print("          workshop (BF16 math)     unknown: math not counted, results are optimistic (pass --tflops)")
print(f"Model     {model_name}: {L} layers, hidden {H}, {nh} query heads / {nkv} KV heads x {hd}, "
      f"MLP {inter}, vocab {n(vocab)}{', tied embeddings' if tied else ''}")
print(f"Requests  {a.inp} in / {a.out} out: a request holds ~{avg_ctx:,.0f} tokens on average "
      f"while decoding, {peak_ctx} at the end")
print()
print(f"Step 1  weights carried per pass   {matmul_params/1e9:.3f} B params x {a.weight_bytes:g} B = {gb(w_read)}"
      f"  -> {w_read/B*1e3:.2f} ms on the belt")
print(f"Step 2  KV bytes per token         2 (K,V) x {L} layers x {nkv} KV heads x {hd} x {a.kv_bytes:g} B = {n(kv_tok)} B")
print(f"Step 3  KV bytes per user          {avg_ctx:,.0f} tokens x {n(kv_tok)} B = {mb(kv_user)}")
print(f"Step 4  tokens per pass            users x (1 decode + {prompt_share:g} prompt tokens riding along)")
print(f"Step 5  pass time (TPOT)           max(weights on belt, tokens x {flops_tok/1e9:.2f} GFLOP in workshop)"
      f" + users x {mb(kv_user)} on belt")
if mem_gib:
    print(f"Step 6  memory cap                 ({mem_gib:,.1f} GiB x {a.gpu_util:g} - {gb(w_stored)} weights) / {n(kv_tok)} B "
          f"= {n(max_tokens)} tokens -> {n(c_mem)} requests of {peak_ctx} tokens")
else:
    print(f"Step 6  memory cap                 skipped (pass --mem-gib)")
print(f"Step 7  ceiling                    each output token costs {cost_math*1e6:.1f} us of math + "
      f"{cost_kv*1e6:.1f} us of KV carrying -> at most {n(tok_inf)} tok/s, however many users")
print()

users = [u for u in (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096) if not c_cap or u < c_cap]
if c_cap:
    users.append(c_cap)
print(f"  {'users':>7} {'tokens':>7} {'linear ms':>10} {'(slower)':>9} {'attn ms':>8} {'TPOT ms':>8} "
      f"{'total tok/s':>12} {'per-user tok/s':>15}")
for u in users:
    lin, who, att = pass_parts(u)
    t = lin + att
    mark = f"  <- {cap_why} cap" if (c_cap and u == c_cap) else ""
    print(f"  {n(u):>7} {n(tokens_in_pass(u)):>7} {lin*1e3:>10.2f} {who:>9} {att*1e3:>8.2f} {t*1e3:>8.2f} "
          f"{n(u/t):>12} {n(1/t):>15}{mark}")
print()

if c_cap:
    best = c_cap / pass_time(c_cap)
    print(f"Max output throughput   ~{n(best)} tok/s at {n(c_cap)} users (limited by {cap_why})")
else:
    best = tok_inf
    print(f"Max output throughput   ~{n(best)} tok/s, approached as users grow (no memory figure or cap given)")
print(f"                        ~{best/a.out:,.0f} requests/s  <- open-loop knee "
      f"(send more per second than this and the waiting line grows)")
cs = max_users_under(a.tpot_max / 1000)
if cs:
    ts = pass_time(cs)
    print(f"Under TPOT <= {a.tpot_max:g} ms     ~{n(cs)} users, ~{n(cs/ts)} tok/s ({cs/ts/best:.0%} of max), "
          f"~{cs/ts/a.out:,.0f} requests/s  <- a cap of ~{n(cs)} keeps this promise")
else:
    print(f"Under TPOT <= {a.tpot_max:g} ms     not reachable even at 1 user ({pass_time(1)*1e3:.2f} ms)")
print()
print("Not modeled: time before the first token (HTTP, tokenizing), random bunching of arrivals,")
print("multi-GPU, MoE, sliding-window attention. Efficiencies were calibrated on one GPU and model;")
print("pass --calibrate with your own 1-user TPOT for a better fit.")
print()
PY
