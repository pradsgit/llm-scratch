import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int,
        eps: float=1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # init gain parameter of shape d_model
        self.gain = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """process a tensor of shape (batch_size, seq_len, d_model) and return a tensor of the same shape."""
        x_dtype = x.dtype
        # upcast to float32 to prevent overflow when sqaured
        x = x.to(torch.float32)
        # calculate rms
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        output = (x / rms) * self.gain.to(torch.float32)
        return output.to(x_dtype)