import torch
from layer import LayerCache
from paged import PagedSeq

class Request:
    def __init__(self, rid, prompt, max_new, arrival=0):
        self.rid = rid                # a name, for printing
        self.prompt = prompt          # list of token ids
        self.max_new = max_new        # stop after this many new tokens
        self.arrival = arrival        # which pass it shows up at
        self.out = []                 # tokens generated so far
        self.caches = None            # 28 LayerCaches, created on admission
        self.feed = None              # tokens to push in the next pass
        self.start = 0                # tokens already in its cache = position of feed[0]
        self.done = False


@torch.no_grad()
def serve(model, requests, max_batch, eos_id=None):
    waiting = sorted(requests, key=lambda r: r.arrival)
    running, finished = [], []
    step = 0

    while waiting or running:
        # 1. ADMIT — while waiting[0] has arrived (arrival <= step)
        #    and len(running) < max_batch:
        #        r = waiting.pop(0)
        #        r.caches = [LayerCache() for _ in model.layers]
        #        r.feed = list(r.prompt);  r.start = 0
        #        running.append(r)
        while len(running) < max_batch and waiting and waiting[0].arrival <= step:
            r = waiting.pop(0)
            r.caches = [LayerCache() for _ in model.layers]
            r.feed = list(r.prompt)
            r.start = 0
            running.append(r)

        # 2. if nothing is running yet (no one has arrived), step += 1 and continue
        if not running:
            step += 1
            continue

        # 3. BUILD THE PILE — loop over running, track an offset:
        #        n = len(r.feed)
        #        seqs.append((off, n));  off += n
        #        positions += range(r.start, r.start + n)
        #        ids += r.feed
        ids, positions, seqs = [], [], []
        off = 0
        for r in running:
            n = len(r.feed)
            seqs.append((off, n))
            off += n
            positions.extend(range(r.start, r.start + n))
            ids.extend(r.feed)

        print(f"pass {step:>2}: " + "  ".join(f"{r.rid}×{len(r.feed)}" for r in running))

        # 4. RUN — logits = model.forward_batch(
        #              torch.tensor(ids), torch.tensor(positions),
        #              seqs, [r.caches for r in running])        → (U, 151936)
        logits = model.forward_batch(
            torch.tensor(ids), torch.tensor(positions),
            seqs, [r.caches for r in running])

        # 5. UPDATE each user — for i, r in enumerate(running):
        #        t = logits[i].argmax().item()
        #        r.out.append(t)
        #        r.start += len(r.feed)        ← everything fed is now cached
        #        r.feed = [t]                  ← next pass feeds just the new token
        #        if t == eos_id or len(r.out) >= r.max_new: r.done = True
        for i, r in enumerate(running):
            t = logits[i].argmax().item()
            r.out.append(t)
            r.start += len(r.feed)
            r.feed = [t]
            if t == eos_id or len(r.out) >= r.max_new:
                r.done = True

        # 6. REMOVE finished — move done ones to finished, drop their caches
        #    (r.caches = None) so the memory is freed
        for r in running[:]:
            if r.done:
                finished.append(r)
                running.remove(r)
                r.caches = None

        step += 1
    return finished


@torch.no_grad()
def serve_paged(model, requests, pool, max_batch, eos_id=None):
    bs = pool.block_size
    total = len(pool.free)              # every block in the pool
    promised = 0                        # worst-case blocks promised to running requests

    waiting = sorted(requests, key=lambda r: r.arrival)
    running, finished = [], []
    step = 0

    while waiting or running:
        # 1. ADMIT — arrived, a batch slot free, AND its worst case fits in memory
        while len(running) < max_batch and waiting and waiting[0].arrival <= step:
            r = waiting[0]
            need = (len(r.prompt) + r.max_new + bs - 1) // bs   # worst case, in blocks
            if promised + need > total:
                break                                           # no room yet: it waits
            waiting.pop(0)
            promised += need
            r.need = need
            r.seq = PagedSeq(pool)
            r.caches = [r.seq.layer(l) for l in range(len(model.layers))]
            r.feed = list(r.prompt)
            r.start = 0
            running.append(r)

        # nobody running: either no one has arrived yet, or the next one can never fit
        if not running:
            if waiting and waiting[0].arrival <= step:
                raise RuntimeError(f"request {waiting[0].rid} needs more blocks than the pool has")
            step += 1
            continue

        # 2. RESERVE — make sure each request owns blocks for this pass's new tokens
        for r in running:
            r.seq.reserve(len(r.feed))

        # 3. BUILD THE PILE
        ids, positions, seqs = [], [], []
        off = 0
        for r in running:
            n = len(r.feed)
            seqs.append((off, n))
            off += n
            positions.extend(range(r.start, r.start + n))
            ids.extend(r.feed)

        print(f"pass {step:>2}: " + "  ".join(f"{r.rid}×{len(r.feed)}" for r in running)
              + f"   free={len(pool.free)}  waiting={sum(w.arrival <= step for w in waiting)}")

        # 4. RUN
        logits = model.forward_batch(
            torch.tensor(ids), torch.tensor(positions),
            seqs, [r.caches for r in running])

        # 5. UPDATE each request
        for i, r in enumerate(running):
            t = logits[i].argmax().item()
            r.out.append(t)
            r.seq.commit(len(r.feed))     # tokens are now stored in every layer's pool
            r.start += len(r.feed)
            r.feed = [t]
            if t == eos_id or len(r.out) >= r.max_new:
                r.done = True

        # 6. REMOVE finished — give their blocks back to the pool
        for r in running[:]:
            if r.done:
                r.seq.release()
                promised -= r.need
                r.caches = None
                running.remove(r)
                finished.append(r)

        step += 1
    return finished