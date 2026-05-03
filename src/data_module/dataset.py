import torch
from torch.utils.data import Dataset

from nets.embeddingLayers.imageEmbed import ImageEmbed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImgDataset(Dataset):
    def __init__(self, ds):
      self.ds = ds
      self.img_embed = ImageEmbed().to(device)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        img = self.img_embed(img)

        label = self.ds[idx]["text"]
        return img, label
