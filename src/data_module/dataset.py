import torch
from torch.utils.data import Dataset

from nets.embeddingLayers.imageEmbed import ImageEmbed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImgDataset(Dataset):
    def __init__(self, ds, img_label, text_label):
      self.ds = ds
      self.img_label = img_label
      self.text_label = text_label
      self.img_embed = ImageEmbed().to(device)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx][self.img_label] 
        img = self.img_embed(img)

        label = self.ds[idx][self.text_label]
        return img, label
