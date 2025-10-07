# implements a Linear and Embedding modules from scratch
import math
import torch
import torch.nn as nn
    
class Embedding(nn.Module):
    """implements embedding lookup"""
    def __init__(
        self, 
        num_embeddings: int, 
        embed_dim: int, 
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embed_dim, dtype=dtype, device=device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        """initialize self.weight parameter with values drawn from a truncated normal distribution."""
        a = -3
        b = 3
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.weight, a=a, b=b)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """lookup each ID in input to corresponsd word emebdding"""
        # check if the input is tensor
        assert isinstance(input, torch.Tensor), "input should of type Tensor"
            
        return self.weight[input]


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
    

if __name__ == "__main__":
    num_samples = 20
    in_features = 10
    out_features = 64
    input = torch.rand((num_samples, in_features))