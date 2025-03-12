import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
  """
  SinusoidalPositionalEncoding computes the sinusoidal positional encodings.

  Attributes:
    max_len (int): Maximum number of positions per row
    d_model (int): Dimension of models vector embedding
  """

  def __init__(self, max_len: int, d_model: str):
    super().__init__()

    self.pe = torch.zeros(max_len, d_model) # same shape as input matrix
    self.pe.requires_grad = False # no need to compute gradient

    position = torch.arange(max_len, dtype=torch.float).unsqueeze(dim=1) # shape (max_len, 1)
    _2i = torch.arange(0, d_model, step=2, dtype=torch.float) # i is index of model

    self.pe[:, 0::2] = torch.sin(position / 10000 ** (_2i / d_model))
    self.pe[:, 1::2] = torch.cos(position / 10000 ** (_2i / d_model))


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Computing sinusoidal positional encoding based on input shape

    Args:
      x (torch.Tensor): Input matrix of shape (batch_size, seq_len, d_model)
    """

    seq_len = x.shape[1]
    return self.pe[:seq_len, :]