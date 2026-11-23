import statistics
WEIGHTS_GB = 1.19        # all weights, delivered once per pass, shared
KV_PER_USER_GB = 0.022   # one user's K and V (~192 tokens x 114,688 bytes)
BELT_GBPS = 348          # what the A40 actually delivers (half of the 696 spec)
# # WEIGHTS_GB = 4.0
# # KV_PER_USER_GB = 0.010
# # BELT_GBPS = 400

# def pass_time_ms(users):
#     # your code: GB on the belt, then turn it into milliseconds
#     # 1. total GB = weights + users * KV_PER_USER_GB
#     total_gb = WEIGHTS_GB + users * KV_PER_USER_GB
#     # 2. time in seconds = total GB / BELT_GBPS
#     time_s = total_gb / BELT_GBPS
#     # 3. time in milliseconds = time in seconds * 1000
#     time_ms = time_s * 1000
#     return time_ms
    

GFLOP_PER_TOKEN = 1.19   # linear-layer math per token: 2 x 596M weights
WORKSHOP_TFLOPS = 50     # what the A40 achieves here (~1/3 of its 150 spec)
IN_TOKENS = 128


def pass_time_ms(users, new=0):
    # decode tokens + prompt tokens of the newcomers
    tokens = users + new * IN_TOKENS 
    # belt carries the weights                   
    weights_s = WEIGHTS_GB / BELT_GBPS    
    # workshop does the math          
    math_s = tokens * GFLOP_PER_TOKEN / (WORKSHOP_TFLOPS * 1000)  
    # the linear layers take whichever is slower 
    linear_s = max(weights_s, math_s)
    # the belt carries everyone's KV                          
    attention_s = users * KV_PER_USER_GB / BELT_GBPS
    # both steps, one after the other, in ms
    return 1000 * (linear_s + attention_s)                                

# check against your real A40 runs (closed loop, median TPOT)
measured = {1: 3.40, 4: 3.70, 16: 4.52, 64: 7.91}
for users, real in measured.items():
    pred = pass_time_ms(users)
    print(f"{users:>3} users   predicted {pred:5.2f} ms   measured {real:5.2f} ms   off by {(pred - real) / real:+.0%}")
    
print()
print(f"{'users':>6} {'pass ms':>8} {'total tok/s':>12} {'per-user tok/s':>15}")
for users in [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    t_ms = pass_time_ms(users)
    total_tok_s = users / (t_ms / 1000)
    per_user_tok_s = 1 / (t_ms / 1000)  
    print(f"{users:>6} {t_ms:>8.2f} {total_tok_s:>12,.0f} {per_user_tok_s:>15,.0f}")
    
OUT_TOKENS = 128

def closed_loop(n_users, total_requests):
    now_ms = 0.0
    running = []       # requests currently being served, one per user
    finished = []
    sent = 0

    # every user sends their first request at time 0
    for _ in range(n_users):
        running.append({"sent_at": now_ms, "left": OUT_TOKENS})
        sent += 1

    while running:
        # 1. one pass happens: the clock moves forward by how long this pass takes
        # now_ms += ...                       # your code
        now_ms += pass_time_ms(len(running))

        # 2. every running request gets one token
        for r in running:
            r["left"] -= 1
            if r["left"] == OUT_TOKENS - 1:
                r["first_token_at"] = now_ms
            

        # 3. finished requests leave, and each one's user sends their next request right away
        done = [r for r in running if r["left"] == 0]
        for r in done:
            running.remove(r)
            r["ended_at"] = now_ms
            finished.append(r)
            if sent < total_requests:
                # your code: add the new request to running
                running.append({"sent_at": now_ms, "left": OUT_TOKENS})
                sent += 1

    duration_s = now_ms / 1000
    req_per_s = len(finished) / duration_s
    tok_per_s = req_per_s * OUT_TOKENS
    avg_stay_s = sum(r["ended_at"] - r["sent_at"] for r in finished) / len(finished) / 1000
    avg_first_token_ms = sum(r["first_token_at"] - r["sent_at"] for r in finished) / len(finished)
    return req_per_s, tok_per_s, avg_stay_s, avg_first_token_ms


print()
real_tok_s = {1: 276, 4: 968, 16: 2992, 64: 6965}   # Thursday night, A40
print(f"{'users':>6} {'req/s':>7} {'tok/s':>8} {'real tok/s':>11} {'stay s':>7} {'req/s x stay':>13} {'first token ms':>15}")
for n in [1, 4, 16, 64]:
    rps, tps, stay, first_token = closed_loop(n, total_requests=n * 10)
    print(f"{n:>6} {rps:>7.1f} {tps:>8,.0f} {real_tok_s[n]:>11,} {stay:>7.3f} {rps * stay:>13.1f} {first_token:>15.3f}")
    

def open_loop(rate, duration_s=30, cap=256):
    gap_ms = 1000 / rate                      # your code: time between two arrivals, in ms, at `rate` per second
    arrivals = [i * gap_ms for i in range(int(rate * duration_s))]   # when each request arrives
    waiting, running, finished = [], [], []
    now_ms = 0.0
    next_arrival = 0                   # index of the next request yet to arrive
    max_waiting = 0

    while next_arrival < len(arrivals) or waiting or running:
        # 1. everyone who has arrived by now joins the waiting line
        while next_arrival < len(arrivals) and arrivals[next_arrival] <= now_ms:
            waiting.append({"sent_at": arrivals[next_arrival], "left": OUT_TOKENS})
            next_arrival += 1

        # 2. the scheduler moves people from the front of the line into running, up to the cap
        while waiting and len(running) < cap:
            running.append(waiting.pop(0))     # your code (hint: waiting.pop(0) takes the front of the line)

        max_waiting = max(max_waiting, len(waiting))

        # 3. if nobody is running, jump the clock to the next arrival
        if not running:
            now_ms = arrivals[next_arrival]
            continue

        # 4. one pass: same as the closed loop
        new = sum(1 for r in running if r["left"] == OUT_TOKENS)   # count newcomers for THIS pass
        now_ms += pass_time_ms(len(running), new) 
        for r in running:
            r["left"] -= 1
            if r["left"] == OUT_TOKENS - 1:
                r["first_at"] = now_ms

        # 5. finished requests leave. Nobody re-sends: new requests only come from the clock
        done = [r for r in running if r["left"] == 0]
        for r in done:
            running.remove(r)
            r["ended_at"] = now_ms
            finished.append(r)

    req_per_s = len(finished) / (now_ms / 1000)
    ttft = statistics.median(r["first_at"] - r["sent_at"] for r in finished)
    tpot = statistics.median((r["ended_at"] - r["first_at"]) / (OUT_TOKENS - 1) for r in finished)
    return req_per_s, ttft, tpot, max_waiting


print()
real = {16: (15.7, 34, 4.49), 32: (31.4, 45, 5.87), 48: (47.0, 50, 6.92), 64: (61.8, 75, 13.32),
        96: (68.2, 4715, 28.19), 128: (66.6, 9639, 29.39), 160: (69.9, 10287, 27.83)}   # your A40 sweep
print(f"{'rate':>5} {'req/s':>7} {'real':>6} {'TTFT ms':>8} {'real':>7} {'TPOT':>6} {'real':>6} {'max waiting':>12}")
for rate, (r_rps, r_ttft, r_tpot) in real.items():
    rps, ttft, tpot, maxw = open_loop(rate)
    print(f"{rate:>5} {rps:>7.1f} {r_rps:>6.1f} {ttft:>8.0f} {r_ttft:>7.0f} {tpot:>6.2f} {r_tpot:>6.2f} {maxw:>12}")    
    
    
print("\noverloaded: rate 160")
print(f"{'cap':>5} {'req/s':>7} {'TPOT ms':>8} {'TTFT ms':>8} {'max waiting':>12}")
for cap in [64, 128, 256, 512, 1024]:
    rps, ttft, tpot, maxw = open_loop(160, cap=cap)
    print(f"{cap:>5} {rps:>7.1f} {tpot:>8.2f} {ttft:>8.0f} {maxw:>12}")

print("\nnormal load: rate 48")
for cap in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    rps, ttft, tpot, maxw = open_loop(48, cap=cap)
    print(f"{cap:>5} {rps:>7.1f} {tpot:>8.2f} {ttft:>8.0f} {maxw:>12}")