import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
  def __init__(self, hidden_size: int, epsilon: float = 10**-5):
    super().__init__()
    self.hidden_size = hidden_size
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(hidden_size)) # multiplicative
    self.beta = nn.Parameter(torch.zeros(hidden_size)) # additive

  def forward(self, x):
    # x: (batch, seq_len, hidden_size)
    mu = torch.mean(x, dim=-1, keepdim=True) # (batch, seq_len, 1)
    sigma = torch.std(x, dim=-1, keepdim=True, unbiased=False) # (batch, seq_len, 1)
    x_norm = (x - mu) / (sigma + self.epsilon)
    # gamma and beta are (hidden_size,) -> unsqueeze to become broadcastable
    return self.gamma.unsqueeze(0).unsqueeze(0) * x_norm + self.beta.unsqueeze(0).unsqueeze(0)
