import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer

from src.nets.embeddingLayers.embeddingProjector import EmbeddingProjector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextEmbed(nn.Module):
  def __init__(self, out_dim):
    super().__init__()
    model_name = "google-t5/t5-small"

    self.T5 = T5EncoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    self.T5.requires_grad_(False)
    self.T5.eval()

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    self.embed_proj = EmbeddingProjector(512, out_dim) # hardcoded, 512 is T5 output dim

  def forward(self, input_text):
    inputs = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device) # Added return_tensors='pt'

    with torch.no_grad():
      embeddings = self.T5(**inputs).last_hidden_state.float() # float converts f16 -> f32

    embed_out = self.embed_proj(embeddings)
    return embed_out