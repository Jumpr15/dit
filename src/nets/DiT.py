import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import diffusers
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from huggingface_hub import PyTorchModelHubMixin



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DIT(L.LightningModule, PyTorchModelHubMixin):
  def __init__(
      self,
      batch_size,
      patch_size,
      out_channels,
      embed_dims,
      head_size,
      num_heads,
      block_num,
      lr,
      iterations
  ):
      super().__init__()

      self.batch_size = batch_size
      self.patch_size = patch_size
      self.out_channels = out_channels
      self.lr = lr
      self.iterations = iterations

      self.block_list = nn.ModuleList(
          [DiT_Block(embed_dims, head_size, num_heads) for _ in range(block_num)]
      )
      self.text_embedder = TextEmbed(embed_dims)
      self.timestep_embedder = TimestepEmbed(embed_dims)
      self.final_layer = FinalLayer(embed_dims, patch_size, out_channels)

      self.scheduler = diffusers.schedulers.DDPMScheduler()

  def configure_optimizers(self):
      params = [p for parameters in self.parameters() if p.requires_grad_]

      optimizer = optim.AdamW(params, lr=self.lr)
      scheduler = optim.lr_scheduler.OneCycleLR(
          optimizer,
          self.lr,
          total_steps=self.iterations,
          pct_start=0.1,
          anneal_strategy="cos",
          final_div_factor=100.0
      )

      return {
          "optimizer": optimizer,
          "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
      }

  def unpatchify(self, x):
      # h = w = int(x.shape[1] ** 0.5)
      h = 32
      w = 16
      # CHANGE THIS

      x = torch.reshape(x, (self.batch_size, h, w, self.patch_size, self.patch_size, self.out_channels))
      x = torch.einsum('nhwpqc->nchpwq', x)
      x = torch.reshape(x, (self.batch_size, self.out_channels, h*self.patch_size, w*self.patch_size))

      return x

  def forward(self, latent_img, text, timestep):
      text_embed = self.text_embedder(text)
      timestep_embed = self.timestep_embedder(timestep)

      for block in self.block_list:
          latent_img = block(latent_img, text_embed, timestep_embed)

      final_out = self.final_layer(block, timestep)
      out = self.unpatchify(final_out)

      return out

  def training_step(self, latent_img, text_label):
      noise = torch.randn_like(latent_img)
      steps = torch.randint(self.scheduler.config.num_train_timesteps, (self.batch_size, )).to(device)
      noised_latents = self.scheduler.add_noise(latent_img, noise, steps)
      out = self(noised_latents, steps)
      loss = F.mse_loss(out, noise)
      return loss




