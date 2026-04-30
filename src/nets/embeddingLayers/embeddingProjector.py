import torch.nn as nn

class EmbeddingProjector(nn.Module):
  def __init__(self, embed_dim, out_dim, exp_factor=4):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, embed_dim*exp_factor),
        nn.SiLU(),
        nn.Linear(embed_dim*exp_factor, out_dim)
    )

  def forward(self, x):
    return self.mlp(x)
