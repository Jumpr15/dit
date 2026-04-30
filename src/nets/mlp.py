import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, embed_dims, exp_factor=4):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(embed_dims, embed_dims*exp_factor),
        nn.SiLU(),
        nn.Linear(embed_dims*exp_factor, embed_dims),
        nn.LayerNorm(embed_dims)
    )

  def forward(self, x):
    return self.mlp(x)