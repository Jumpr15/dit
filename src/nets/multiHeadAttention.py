import torch.nn as nn
import torch.nn.functional as F

class Multi_Head_Attention(nn.Module):
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

  def forward(self, q_logits, k_logits, v_logits):

    batch_size = q_logits.shape[0]
    q_seq_len = q_logits.shape[1]
    k_seq_len = k_logits.shape[1] # k and v logits should be from same source
    v_seq_len = v_logits.shape[1]
    
    assert k_seq_len == v_seq_len
    
    kv_seq_len = k_seq_len

    q = self.q_proj(q_logits).view(batch_size, q_seq_len, self.num_heads, self.head_size).transpose(1, 2)
    k = self.k_proj(k_logits).view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(1, 2)
    v = self.v_proj(v_logits).view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(1, 2)

    attention_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.total_heads)

    return self.o_proj(out)