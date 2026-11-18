import torch

class BlockPool:
    def __init__(self, num_layers, num_blocks, block_size, n_kv, head_dim):
        self.block_size = block_size
        # one K and one V tensor per layer, shape (num_blocks, block_size, n_kv, head_dim)
        # self.K = [torch.zeros(...) for _ in range(num_layers)]
        self.K = [torch.zeros((num_blocks, block_size, n_kv, head_dim)) for _ in range(num_layers)]
        # self.V = [same]
        self.V = [torch.zeros((num_blocks, block_size, n_kv, head_dim)) for _ in range(num_layers)]
        # self.free = list(range(num_blocks))         ← every block starts free
        self.free = list(range(num_blocks))

    def allocate(self):
        # if self.free is empty: raise RuntimeError("out of KV blocks")
        # return self.free.pop()
        if not self.free:
            raise RuntimeError("out of KV blocks")
        return self.free.pop()

    def release(self, blocks):
        # self.free.extend(blocks)
        self.free.extend(blocks)

class PagedSeq:
    def __init__(self, pool):
        self.pool = pool
        self.table = []          # this request's blocks, in order
        self.length = 0          # tokens already stored

    def reserve(self, n):
        # make sure the table has enough blocks for self.length + n tokens
        # blocks needed = ceil((self.length + n) / block_size)
        bs = self.pool.block_size
        needed = (self.length + n + bs - 1) // bs
        #   — in integers: (self.length + n + bs - 1) // bs
        # while len(self.table) < needed: self.table.append(self.pool.allocate())
        while len(self.table) < needed:
            self.table.append(self.pool.allocate())

    def commit(self, n):
        # self.length += n        ← once per pass, after all 28 layers
        self.length += n

    def release(self):
        # give every block back to the pool, empty the table
        self.pool.release(self.table)
        self.table = []
        self.length = 0

    def layer(self, l):
        return PagedLayer(self, l)

class PagedLayer:
    def __init__(self, seq, l):
        self.seq, self.l = seq, l

    def append(self, k, v):
        seq, pool, bs = self.seq, self.seq.pool, self.seq.pool.block_size
        n = k.shape[2]

        # WRITE — the n new tokens sit at positions length … length+n-1
        pos = torch.arange(seq.length, seq.length + n)
        blk = torch.tensor(seq.table)[pos // bs]     # chunk → physical block
        slot = pos % bs
        pool.K[self.l][blk, slot] = k[0].transpose(0, 1)   # (n, 8, 128)
        pool.V[self.l][blk, slot] = v[0].transpose(0, 1)

        # READ — all L tokens, in order
        L = seq.length + n
        # k_all = pool.K[self.l][seq.table].reshape(-1, n_kv, head_dim)[:L]   → (L, 8, 128)
        k_all = pool.K[self.l][torch.tensor(seq.table)].reshape(-1, k.shape[1], k.shape[3])[:L]
        # same for v
        v_all = pool.V[self.l][torch.tensor(seq.table)].reshape(-1, v.shape[1], v.shape[3])[:L]
        # return k_all.transpose(0, 1)[None], v_all.transpose(0, 1)[None]    → (1, 8, L, 128)
        return k_all.transpose(0, 1)[None], v_all.transpose(0, 1)[None]