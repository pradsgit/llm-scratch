import torch
import torch.nn as nn
from linear import Linear
import math


def silu(x: torch.Tensor) -> torch.Tensor:
    """implements SiLU: input is multiplied by its sigmoid"""
    return x * torch.sigmoid(x)

class PositionWiseFeedForward(nn.Module):
    """implements SwiGLU ffn"""
    def __init__(
        self, 
        d_model: int, 
        d_ff: int | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(math.ceil((8*d_model // 3) / 64) * 64)
        self.layer1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.layer2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.layer3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input x has a shape of (batch_sz, seq_len, d_model)"""
        #TODO: can be replaced with GeGELU
        gate = silu(self.layer1(x)) # gate shape is (b, t, d_ff)
        value = self.layer3(x) # value shape is (b, t, d_ff)
        # Gated Linear Unit GLU, element-wise multiplication
        output = gate * value
        output = self.layer2(output)
        return output
