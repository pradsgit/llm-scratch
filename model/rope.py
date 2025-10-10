import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_k = d_k

        assert d_k % 2 == 0, "dimension should be even in RotaryPositionEmbedding"
        
        inv_freqs = 1.0 / (theta**((torch.arange(0, d_k, 2, device=device).float() - 1) / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freqs) # freqs shape is [max_seq_len, d_k//2]

        self.register_buffer('freq_cos', freqs.cos(), persistent=False)
        self.register_buffer('freq_sin', freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x shape is (..., seq_len, d_k); return shape is same
        # token_positions shape is (..., seq_len) specifying the token positions of x along the sequence dimension.

        cos = self.freq_cos[token_positions]
        sin = self.freq_sin[token_positions]

        # even pairs
        x1 = x[..., ::2]
        # odd pairs
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.flatten(-2)


