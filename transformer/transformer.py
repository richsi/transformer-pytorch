import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from positional_encoding import SinusoidalPositionalEncoding

class Transformer(nn.Module):
  def __init__(
    self,
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    num_layers: int = 6,
    dropout: float = 0.1
  ):
    super().__init__()
    
    self.d_model = d_model
    self.num_layers = num_layers

    # embedding layers for source and target

    self.pos_enc = SinusoidalPositionalEncoding(src_vocab_size ,d_model, dropout)
    self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)] 
                                 for _ in range(num_layers))
    self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)] 
                                 for _ in range(num_layers))

    self.linear = nn.Linear(d_model, tgt_vocab_size)
    self.softmax = nn.Softmax()

  def foward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None):
    batch_size, src_len = src_seq.size()
    batch_size, tgt_len = tgt_seq.size()