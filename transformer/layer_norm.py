import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
  def __init__(self, epsilon: float = 10**-6):
    super().__init__()
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(1)) # multiplicative
    self.bias = nn.Parameter(torch.zeros(1)) # additive

  def forward(self, x):
    # x: (batch, seq_len, hidden_size)
    mu = torch.mean(x, dim=-1, keepdim=True) # (batch, seq_len, 1)
    std = torch.std(x, dim=-1, keepdim=True) # (batch, seq_len, 1)
    return self.gamma * (x - mu) / (std + self.eps) + self.bias
