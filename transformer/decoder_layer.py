import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForwardNet
from layer_norm import LayerNormalization

class DecoderLayer(nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
    super().__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    self.enc_dec_mha = MultiHeadAttention(d_model, num_heads, dropout)
    self.ffn = FeedForwardNet(d_model, d_ff, dropout)
    self.norm1 = LayerNormalization(d_model)
    self.norm2 = LayerNormalization(d_model)
    self.norm3 = LayerNormalization(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
    attn_out = self.mha(x, x, x, tgt_mask) # shape: (batch, seq_len, d_model)
    x = x + self.dropout(attn_out)
    x = self.norm1(x)

    attn_out2 = self.enc_dec_mha(x, enc_output, enc_output, memory_mask)
    x = x + self.dropout(attn_out2)
    x = self.norm2(x)

    ffn_out = self.ffn(x)
    x = x + self.dropout(ffn_out)
    x = self.norm3(x)

    return x