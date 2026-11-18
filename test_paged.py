import torch
from layer import LayerCache
from paged import BlockPool, PagedSeq

pool = BlockPool(num_layers=1, num_blocks=8, block_size=4, n_kv=8, head_dim=128)
seq = PagedSeq(pool)
paged = seq.layer(0)
old = LayerCache()

for n in [6, 1, 1]:                         # prefill 6 tokens, then two decodes
    k = torch.randn(1, 8, n, 128)
    v = torch.randn(1, 8, n, 128)
    seq.reserve(n)
    pk, pv = paged.append(k, v)
    ok, ov = old.append(k, v)
    seq.commit(n)
    print(f"n={n}  L={seq.length}  table={seq.table}  "
          f"k diff={(pk - ok).abs().max().item()}  v diff={(pv - ov).abs().max().item()}")

print("free blocks:", len(pool.free))
seq.release()
print("after release:", len(pool.free))