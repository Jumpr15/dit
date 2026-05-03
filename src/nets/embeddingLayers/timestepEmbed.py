from diffusers.models.embeddings import Timesteps, TimestepEmbedding
import torch.nn as nn

class TimestepEmbed(nn.Module):
  def __init__(self, out_dim):
    super().__init__()

    # mostly hardcoded
    in_c = 256

    self.time_proj = Timesteps(
    num_channels=in_c,
    flip_sin_to_cos=True,
    downscale_freq_shift=0
    ) # [B, 1] => [B, num_channels]

    self.time_embed_dit_block = TimestepEmbedding(
    in_channels=in_c,
    time_embed_dim=out_dim*6
    ) # [B, num_channels] => [B, time_embed_dims]
    
    self.time_embed_final = TimestepEmbedding(
      in_channels=in_c,
      time_embed_dim=out_dim*2
    )

    self.initialize_weights()

  def initialize_weights(self):
    # timesteps are linear 1 + b1 => linear 2 + b2
    nn.init.normal_(self.time_embed_dit_block.linear_1.weight, std=0.02)
    nn.init.constant_(self.time_embed_dit_block.linear_1.bias, 0)
    nn.init.normal_(self.time_embed_dit_block.linear_2.weight, std=0.02)
    nn.init.constant_(self.time_embed_dit_block.linear_2.bias, 0)

    nn.init.normal_(self.time_embed_final.linear_1.weight, std=0.02)
    nn.init.constant_(self.time_embed_final.linear_1.bias, 0)
    nn.init.normal_(self.time_embed_final.linear_2.weight, std=0.02)
    nn.init.constant_(self.time_embed_final.linear_2.bias, 0)

  def forward(self, t):
    proj_t = self.time_proj(t)
    t_block_out = self.time_embed_dit_block(proj_t)
    t_final_out = self.time_embed_final(proj_t)
    return t_block_out, t_final_out
