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