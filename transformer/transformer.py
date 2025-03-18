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
    n_heads: int = 8,
    d_ff: int = 2048,
    num_layers: int = 6,
    dropout: float = 0.1,
    tie_weights: bool = True
  ):
    super().__init__()
    
    self.d_model = d_model
    self.num_layers = num_layers