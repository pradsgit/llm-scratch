# implements scaled dot product attention SDPA

import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
from rope import RotaryPositionEmbedding

# write a softmax function to apply on a tensor along dim dimension
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # calculate max value along the dim on which softmax is applied
    max = torch.max(x, dim=dim, keepdim=True)[0]
    exp = torch.exp(x - max) # for numerical stability
    sum = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None, # boolean mask
    scale: float | None = None,
):
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    scores = einsum(
        query, key, 
        "batch ... seq_q d_k, batch ... seq_k d_k -> batch ... seq_q seq_k"
    ) * scale_factor

    # scores = (query @ key.transpose(-2, -1)) * scale_factor
    if mask is not None:
        scores.masked_fill_(~mask, float('-inf'))
    
    attn_weights = softmax(scores, dim=-1)
    output = einsum(
        attn_weights, value,
        "batch ... seq_q seq_k, batch ... seq_k d_v -> batch ... seq_q d_v"
    )
    # output = attn_weights @ value
    return output


# multi-head self-attention layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(10000, self.d_k, seq_len, device)

        # also calculate causal mask
        self.register_buffer('causal_mask', torch.tril(torch.ones(seq_len, seq_len, device=device).view(1, 1, seq_len, seq_len)))

    def forward(self, x: torch.Tensor):
        # x shape is (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        assert self.d_model == d_model

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
            .permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, seq_len, d_k)
        )

        q, k, v = qkv.unbind(0) # q, k, v shape is (batch_size, num_heads, seq_len, d_k)

        # apply RoPE to q and k
        token_positions = torch.arange(seq_len, device=x.device)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # calculate sdpa
        y = scaled_dot_product_attention(q, k, v, self.causal_mask) 
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        y = self.c_proj(y) # y shape is (batch_size, seq_len, d_model)
        return y

        



