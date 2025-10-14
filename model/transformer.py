# implements a Linear and Embedding modules from scratch
import math
import torch
import torch.nn as nn
from linear import Linear
from attention import MultiHeadSelfAttention, softmax
from embedding import Embedding
from ffn import PositionWiseFeedForward
from norm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int, 
        d_ff: int,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.multihead_attention = MultiHeadSelfAttention(d_model, num_heads, seq_len, device)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device, dtype)
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.multihead_attention(self.rms_norm(x))
        x = x + self.ffn(self.rms_norm(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int, 
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        # build layers of TransformerBlock
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, context_length) for _ in num_layers]
        )
        self.rms_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # x shape is (batch_size, seq_len)
        x = self.embed(x)
        # x shape is (batch_size, seq_len, d_model)
        x = self.transformer_layers(x)
        # pass it thru an LM head with pre-rms_norm
        logits = self.lm_head(self.rms_norm(x))
        return logits

# if __name__ == "__main__":
#     num_samples = 20
#     in_features = 10
#     out_features = 64
#     input = torch.rand((num_samples, in_features))