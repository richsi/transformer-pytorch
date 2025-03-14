import torch
import torch.nn as nn
from typing import Optional
import math

class SinusoidalPositionalEncoding(nn.Module):
  """
  Attributes:
    max_len (int): Maximum number of positions per row
    d_model (int): Dimension of models vector embedding
  """

  def __init__(self, seq_len: int, d_model: str, dropout: Optional[float] = 0.0):
    super().__init__()

    self.dropout = nn.Dropout(dropout)

    self.pe = torch.zeros(seq_len, d_model) # same shape as input matrix
    self.pe.requires_grad = False

    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=1) # shape (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # for numerical stability

    self.pe[:, 0::2] = torch.sin(position * div_term) 
    self.pe[:, 1::2] = torch.cos(position * div_term)

    self.pe = self.pe.unsqueeze(dim=0) # (1, seq_len, d_model) batching
    # self.register_buffer('pe', pe)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x (torch.Tensor): Input matrix of shape (batch_size, seq_len, d_model)
    """

    seq_len = x.shape[1]
    x = x + self.pe[:, :seq_len, :]
    return self.dropout(x)