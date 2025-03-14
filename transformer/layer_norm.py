import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
  def __init__(self, epsilon: float = 10**-6):
    super().__init__()
    self.epsilon = epsilon
    self.alpha = nn.Parameter(torch.ones(1)) # multiplicative
    self.bias = nn.Parameter(torch.zeros(1)) # additive

  def forward(self, x):
    pass