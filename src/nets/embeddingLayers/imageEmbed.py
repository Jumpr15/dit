import torch
import torch.nn as nn
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageEmbed(nn.Module):
  def __init__(self, in_c, patch_size):
    super().__init__()
    
    self.patch_size = self.patch_size
    self.img_transform = transforms.Compose([
      transforms.v2.RGB(),
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float32)
    ])

    self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)
    self.vae.requires_grad_(False)
    self.vae.eval()

    self.patchify = nn.Sequential(
        nn.Conv2d(
          in_channels=in_c,
          out_channels=in_c*patch_size**2,
          kernel_size=patch_size,
          stride=patch_size
        ),
        nn.Flatten(2)
    )

  def forward(self, img):
    t_img = self.img_transform(img).to(device)
    self.h_patch = t_img.size(dim=1) / self.patch_size
    self.w_patch = t_img.size(dim=2) / self.patch_size
    t_img = t_img.unsqueeze(0)

    with torch.no_grad():
      l_img = self.vae.encode(t_img)
      l_sample = l_img.latent_dist.sample() * 0.18215

    patched_latent = self.patchify(l_sample).transpose(1, 2)
    return patched_latent
  
  def get_hw_patches(self):
    return self.h_patch, self.w_patch