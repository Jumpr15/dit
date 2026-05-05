import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import diffusers
import pytorch_lightning as L
from diffusers.models import AutoencoderKL
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from huggingface_hub import PyTorchModelHubMixin
import PIL.Image as Image

from ema_pytorch import EMA

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
        iterations,
        latent_h,
        latent_w,
        vae,
        vae_scale_factor,
        t_channels=256
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.in_dims = in_dims
        self.embed_dims = embed_dims
        self.lr = lr
        self.iterations = iterations
        
        self.h_patch = latent_h // patch_size
        self.w_patch = latent_w // patch_size
        
        self.up_proj = nn.Linear(in_dims * patch_size**2, embed_dims) # input to block dimension expansion
        
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=embed_dims,
            grid_size=(self.h_patch, self.w_patch),
            output_type="pt"
        )
        self.register_buffer("pos_embed", pos_embed.float()) # f64 => f32
        
        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dims,
                out_channels=in_dims*patch_size**2,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)
        )
        
        self.vae = AutoencoderKL.from_pretrained(vae)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.vae_scale_factor = vae_scale_factor

        self.block_list = nn.ModuleList(
            [DiT_Block(embed_dims, head_size, num_heads) for _ in range(block_num)]
        )
        self.text_embedder = TextEmbed(embed_dims)
        self.timestep_embedder = TimestepEmbed(embed_dims, t_channels=t_channels)
        self.final_layer = FinalLayer(embed_dims, patch_size, out_channels)

        self.scheduler = diffusers.schedulers.DDPMScheduler()
        
        object.__setattr__(self, 'ema_model', EMA(
            self,
            beta=0.9999,
            update_after_step=100,
            update_every=10,
            inv_gamma=1.0,
            power=2/3,
        )) # prevents lightning module recursion
        self.register_buffer("ema_step", torch.tensor(0))

        self.initialize_weights()

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

    def on_fit_start(self):
        self.ema_model.to(device) # move ema to gpu
        
    def on_after_backward(self):
        self.ema_model.update()
        self.ema_step += 1

    def initialize_weights(self):
        # basic xavier weight inits for linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # patchify conv2d layer
        w = self.patchify[0].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchify[0].bias, 0)

        # text embedder, target idx 0 and 2 (linear layers)
        nn.init.normal_(self.text_embedder.embed_proj.mlp[0].weight, std=0.02)
        nn.init.normal_(self.text_embedder.embed_proj.mlp[2].weight, std=0.02)

        for block in self.block_list:
            # dit block list adaln table
            nn.init.zeros_(block.adaLN_scale_table.data[2])
            nn.init.zeros_(block.adaLN_scale_table.data[5])

            # dit block cross attn text condition zeroing
            nn.init.constant_(block.cross_mha.o_proj.weight, 0)
            nn.init.constant_(block.cross_mha.o_proj.bias, 0)
            

        # final layer proj 
        nn.init.constant_(self.final_layer.proj_out.weight, 0)
        nn.init.constant_(self.final_layer.proj_out.bias, 0)
        
        # final layer adaln table
        nn.init.zeros_(self.final_layer.adaLN_scale_table.data)
                

    def unpatchify(self, x):
        b = x.shape[0]
        h = self.h_patch
        w = self.w_patch
        p = self.patch_size
        o = self.out_channels

        x = torch.reshape(x, (b, h, w, p, p, o))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = torch.reshape(x, (b, o, h*p, w*p))

        return x

    def forward(self, latent_img, text_embed, timestep):
        timestep_embed, timestep_embed_final = self.timestep_embedder(timestep)

        x = self.up_proj(latent_img) + self.pos_embed
        for block in self.block_list:
            x = block(x, text_embed, timestep_embed)

        final_out = self.final_layer(x, timestep_embed_final)
        out = self.unpatchify(final_out)

        return out

    def training_step(self, batch, batch_idx):
        img, text_label = batch
        
        with torch.no_grad():
            l_img = self.vae.encode(img.to(device))
            l_sample = l_img.latent_dist.sample() * self.vae_scale_factor
            
        noise = torch.randn_like(l_sample)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (l_sample.shape[0], )).to(device)
        noised_latents = self.scheduler.add_noise(l_sample, noise, steps)
        patched_latent = self.patchify(noised_latents).transpose(1, 2)
        
        text_embed = self.text_embedder(text_label)
        
        out = self(patched_latent, text_embed, steps)
        loss = F.mse_loss(out, noise)
        self.log("train_loss", loss)
        return loss

    def inference(self, text, num_steps=50, use_ema_weights=True):
        self.eval()
        self.scheduler.set_timesteps(num_steps)
        
        with torch.no_grad():
            latent_img = torch.randn(1, 4, 64, 64).to(device) # B, C, L_H, L_W ## changes by image dimensions
            
            for timestep in self.scheduler.timesteps: # 0-num_steps
                timestep = timestep.unsqueeze(0).to(device)
                patched_latent = self.patchify(latent_img).transpose(1, 2)
                
                if use_ema_weights is True:
                    noise_pred = self.ema_model.ema_model(patched_latent, text, timestep) # using ema weights
                else:  
                    noise_pred = self(patched_latent, text, timestep) # predicted noise

                noise_pred_cpu = noise_pred.cpu()
                latent_img_cpu = latent_img.cpu()
                timestep_cpu = timestep.cpu()

                # Step on CPU
                latent_img_cpu = self.scheduler.step(noise_pred_cpu, timestep_cpu, latent_img_cpu).prev_sample

                # Move result back to GPU for next iteration
                latent_img = latent_img_cpu.to(device)
                
            latent_img = latent_img / self.vae_scale_factor # scaling factor matching training
            img = self.vae.decode(latent_img).sample
        
        t_img = ((img * 0.5) + 0.5).clamp(0, 1) # [-1, 1] => [0, 1] (clamp values out of range)
        t_img = t_img.permute(0, 2, 3, 1)
        t_img = t_img * 255 # to rgb vals?
        int_img = t_img.detach().cpu().numpy().astype('uint8') # to uint8
        pil_img = Image.fromarray(int_img[0]) # convert to pil image format
        return pil_img
        