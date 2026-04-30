from diffusers.models.embeddings import Timesteps, TimestepEmbedding

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

    self.time_embed = TimestepEmbedding(
    in_channels=in_c,
    time_embed_dim=out_dim*6
    ) # [B, num_channels] => [B, time_embed_dims]

  def forward(self, t):
    proj_t = self.time_proj(t)
    t_out = self.time_embed(proj_t)
    return t_out
