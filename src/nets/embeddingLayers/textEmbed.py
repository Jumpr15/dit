import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextEmbed(nn.Module):
  def __init__(self, out_dim, exp_factor=4):
    super().__init__()
    model_name = "google-t5/t5-small"

    self.T5 = T5EncoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    self.T5.requires_grad_(False)
    self.T5.eval()

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_out_dim = 512 # output dimension of T5

    self.embed_proj = nn.Sequential(
        nn.Linear(model_out_dim, model_out_dim*exp_factor),
        nn.SiLU(),
        nn.Linear(model_out_dim*exp_factor, out_dim)
    )

  def forward(self, input_text):
    inputs = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device) # Added return_tensors='pt'

    with torch.no_grad():
      embeddings = self.T5(**inputs).last_hidden_state.float() # float converts f16 -> f32

    embed_out = self.embed_proj(embeddings)
    return embed_out