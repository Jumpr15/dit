import torch
from torch.utils.data import Dataset

from nets.embeddingLayers.imageEmbed import ImageEmbed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DigitsDataset(Dataset):
    def __init__(self, ds, out_channels, patch_size):
      self.ds = ds
      self.img_embed = ImageEmbed(patch_size, out_channels).to(device)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["img"]
        l_img = self.img_embed(img)

        label = self.ds[idx]["label"]
        print(label)
        return l_img, label
    
    def __getpatches__(self):
        return self.img_embed.get_hw_patches()