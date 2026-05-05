import torch
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageEmbed(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.img_transform = transforms.Compose([
      # transforms.Resize((512,512)),
      transforms.v2.RGB(),
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float32)
    ])

  def forward(self, img):
    t_img = self.img_transform(img)
    return t_img