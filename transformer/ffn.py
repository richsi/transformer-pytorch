import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    super().__init__()

    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout=dropout)

  def forward(self, x):
    # x shape (batch, seq_len, d_model)
    out = self.linear1(x)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.linear2(out)
    return out