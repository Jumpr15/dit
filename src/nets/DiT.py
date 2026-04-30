import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import diffusers
import lightning as L
from diffusers.models import AutoencoderKL
from huggingface_hub import PyTorchModelHubMixin

from nets.embeddingLayers.textEmbed import TextEmbed
from nets.embeddingLayers.timestepEmbed import TimestepEmbed

from nets.finalLayer import FinalLayer
from nets.ditBlock import DiT_Block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DIT(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        batch_size,
        patch_size,
        out_channels,
        in_dims,
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
        self.in_dims = in_dims
        self.embed_dims = embed_dims
        self.lr = lr
        self.iterations = iterations
        
        self.up_proj = nn.Linear(in_dims * patch_size**2, embed_dims) # input to block dimension expansion
        
        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dims,
                out_channels=in_dims*patch_size**2,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)
        )
        
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.block_list = nn.ModuleList(
            [DiT_Block(embed_dims, head_size, num_heads) for _ in range(block_num)]
        )
        self.text_embedder = TextEmbed(embed_dims)
        self.timestep_embedder = TimestepEmbed(embed_dims)
        self.final_layer = FinalLayer(embed_dims, patch_size, out_channels)

        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]

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
        b = self.batch_size
        h = 1 ####
        w = 1 ####
        p = self.patch_size
        o = self.out_channels

        x = torch.reshape(x, (b, h, w, p, p, o))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = torch.reshape(x, (b, o, h*p, w*p))

        return x

    def forward(self, latent_img, text, timestep):
        text_embed = self.text_embedder(text)
        timestep_embed = self.timestep_embedder(timestep)

        x = self.up_proj(latent_img)
        for block in self.block_list:
            x = block(x, text_embed, timestep_embed)

        final_out = self.final_layer(x, timestep_embed)
        out = self.unpatchify(final_out)

        return out

    def training_step(self, batch, batch_idx):
        img, text_label = batch
        noise = torch.randn_like(img)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (self.batch_size, )).to(device)
        
        with torch.no_grad():
            l_img = self.vae.encode(img)
            l_sample = l_img.latent_dist.sample() * 0.18215
        patched_latent = self.patchify(l_sample).transpose(1, 2)
        
        noised_latents = self.scheduler.add_noise(patched_latent, noise, steps)
        out = self(noised_latents, text_label, steps)
        loss = F.mse_loss(out, noise)
        self.log("train_loss", loss)
        return loss




