import torch
import torch.nn as nn
from timm.models.layers import DropPath

from nets.multiHeadAttention import Multi_Head_Attention

class DiT_Block(nn.Module):
  def __init__(self, embed_dims, head_size, num_heads, exp_factor=4):
    super().__init__()
    self.self_mha = Multi_Head_Attention(embed_dims, head_size, num_heads)
    self.cross_mha = Multi_Head_Attention(embed_dims, head_size, num_heads)
    self.mlp = nn.Sequential(
        nn.Linear(embed_dims, embed_dims*exp_factor),
        nn.SiLU(),
        nn.Linear(embed_dims*exp_factor, embed_dims),
    )
    self.adaLN_scale_table = nn.Parameter(torch.randn(6, embed_dims) / embed_dims ** 0.5)
    self.ln1 = nn.LayerNorm(
        embed_dims,
        elementwise_affine=False,
        eps=1e-6
    )
    self.ln2 = nn.LayerNorm(
        embed_dims,
        elementwise_affine=False,
        eps=1e-6
    )
    self.drop_path = DropPath(0.1)

  def adaLN_modulate(self, x, scale_factor, shift_factor):
    return (x * (1 + scale_factor)) + shift_factor

  def forward(self, x, y, t):
    B = x.shape[0]
    msa_scale, msa_shift, msa_gate, mlp_scale, mlp_shift, mlp_gate = (self.adaLN_scale_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

    s_attn_out = x + self.drop_path(
        msa_gate * self.self_mha(
            self.adaLN_modulate(
                self.ln1(x),
                msa_scale,
                msa_shift
            )
        )
    )

    c_attn_out = s_attn_out + self.drop_path(
        self.cross_mha(s_attn_out, y, y)
    )

    mlp_out = c_attn_out + self.drop_path(
        mlp_gate * self.mlp(
            self.adaLN_modulate(
                self.ln2(c_attn_out),
                mlp_scale,
                mlp_shift
            )
        )
    )

    return mlp_out
