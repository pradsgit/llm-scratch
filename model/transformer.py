# implements a Linear and Embedding modules from scratch
import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    """
    performs a linear transformation: y=xW.T; ignoring bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int, 
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight parameter initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        """initialize self.weight parameter with values drawn from a truncated normal distribution."""
        mean = 0
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        a = -3 * std
        b = 3 * std
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """applies linear transformation operation to incoming data x"""
        return x @ self.weight.T # pytorch implements this in c++ for efficiency
    
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
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embed_dim, dtype=dtype, device=device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        """initialize self.weight parameter with values drawn from a truncated normal distribution."""
        mean = 0
        std = 1
        a = -3
        b = 3
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=a, b=b)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """lookup each ID in input to corresponsd word emebdding"""
        # check if the input is tensor
        assert isinstance(input, torch.Tensor), "input should of type Tensor"
            
        return self.weight[input]




if __name__ == "__main__":
    num_samples = 20
    in_features = 10
    out_features = 64
    input = torch.rand((num_samples, in_features))

    layer = Linear(in_features, out_features)
    output = layer(input)
    print(output.shape)