import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        # initialize params with truncated normal. 

        # Samples from N(0, std**2) but only keeps values in the range [a, b]
        std = math.sqrt(2.0 / self.in_features + self.out_features)
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """applies linear transformation operation to incoming data x"""
        return x @ self.weight.T