import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForwardNet
from layer_norm import LayerNormalization

class EncoderLayer(nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
    super().__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    self.ffn = FeedForwardNet(d_model, d_ff, dropout)
    self.norm1 = LayerNormalization(d_model)
    self.norm2 = LayerNormalization(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    attn_out = self.mha(x, x, x) # shape: (batch, seq_len, d_model)
    x = x + self.dropout(attn_out) # residual
    x = self.norm1(x)

    ff_out = self.ffn(x)
    x = x + self.dropout(ff_out)
    x = self.norm2(x)

    return x