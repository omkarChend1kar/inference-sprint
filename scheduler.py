import torch
from layer import LayerCache

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