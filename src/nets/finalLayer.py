import torch
import torch.nn as nn

class FinalLayer(nn.Module):
  def __init__(self, embed_dims, patch_size, out_channels):
      super().__init__()
      self.ln = nn.LayerNorm(
        embed_dims,
        elementwise_affine=False,
        eps=1e-6
      )
      self.adaLN_scale_table = nn.Parameter(torch.randn(2, embed_dims) / embed_dims ** 0.5)
      self.proj_out = nn.Linear(embed_dims, patch_size*patch_size*out_channels)

  def adaLN_modulate(self, x, scale_factor, shift_factor):
    return (x * (1 + scale_factor)) + shift_factor

  def forward(self, x, t):
    shift, scale = (self.adaLN_scale_table[None] + t.reshape(1, 2, -1)).chunk(2, dim=1)
    x = self.adaLN_modulate(x, scale, shift)
    x = self.proj_out(x)
    return x