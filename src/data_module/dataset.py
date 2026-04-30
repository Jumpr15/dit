import torch
from torch.utils.data import Dataset

from nets.embeddingLayers.imageEmbed import ImageEmbed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DigitsDataset(Dataset):
    def __init__(self, ds):
      self.ds = ds
      self.img_embed = ImageEmbed(4, 1).to(device)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["img"]
        l_img = self.img_embed(img)

        label = self.ds[idx]["label"]
        print(label)
        return l_img, label