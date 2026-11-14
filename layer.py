import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # your code here
        in_dtype = x.dtype # this is to record what came in (float32 CPU, bfloat16 on GPU)
        x = x.to(torch.float32)

        mean_sq = x.pow(2).mean(-1, keepdim=True) # average of x squared, last axis, keepdim=True
        x = x * torch.rsqrt(mean_sq + self.eps) # x times rsqrt(mean_sq + self.eps)

        return x.to(in_dtype) * self.weight 


def rope_tables(head_dim, max_pos, theta=1_000_000.0):
    # returns cos, sin — each (max_pos, head_dim)
    # 64 distinct frequencies, radians per token, per pair
    # basically, 64 distinct sinusoids (rulers), each with a different frequency, that we can use to encode position information into the model's attention mechanism.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    #
    pos = torch.arange(max_pos).float()

    #
    angles = torch.outer(pos, inv_freq)

    #
    emb = torch.cat([angles, angles], dim=-1)

    #
    return torch.cos(emb), torch.sin(emb)


def apply_rope(x, cos, sin):
    # x: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (rigtness / leftness of a pair at that position, up / downness of a pair at that position)
    # returns: (batch, seq_len, num_heads, head_dim)
    # your code here
    seq_len = x.shape[-2]

    # group the last dimension into pairs of numbers (64 pairs for 128 channels)
    half = x.shape[-1] // 2

    # take rows for this sentence, add batch & head axes
    cos = cos[:seq_len][None, None, :, :half]
    sin = sin[:seq_len][None, None, :, :half]

    a = x[..., :half]    # each pair's first number  (channels 0..63)
    b = x[..., half:]    # each pair's second number (channels 64..127)

    new_a = a * cos - b * sin
    new_b = a * sin + b * cos

    return torch.cat([new_a, new_b], dim=-1)