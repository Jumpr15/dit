import torch.nn as nn
import torch.nn.functional as F

class Self_Attention_Head(nn.Module):
  def __init__(self, embed_dims, head_size, num_heads):
    super().__init__()
    self.embed_dims = embed_dims
    self.num_heads = num_heads
    self.head_size = head_size
    self.total_heads = head_size * num_heads

    self.q_proj = nn.Linear(embed_dims, self.total_heads)
    self.k_proj = nn.Linear(embed_dims, self.total_heads)
    self.v_proj = nn.Linear(embed_dims, self.total_heads)
    self.o_proj = nn.Linear(self.total_heads, embed_dims)

  def forward(self, logits):

    batch_size = logits.shape[0]
    seq_len = logits.shape[1]

    q = self.q_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
    k = self.k_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
    v = self.v_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

    attention_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.total_heads)

    return self.o_proj(out)