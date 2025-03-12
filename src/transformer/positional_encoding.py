import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
  def __init__(self, i, d_model)
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    


  def forward(self, x):
    pass