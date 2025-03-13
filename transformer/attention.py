import torch
import torch.nn as nn
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
  """
  Computes num_heads heads of attention.

  Args:
    d_model (int): Total dimension of the model
    num_heads (int): Number of attention heads
    dropout (float): Dropout probability on attention weights
  """

  def __init__(self, d_model, num_heads, dropout=0.1):
    super().__init__()
    assert d_model % num_heads == 0 # d_model must be divisible by num_heads

    self.d_model = d_model 
    self.num_heads = num_heads

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)


  def forward(self, x):
    # x.shape = (batch_size, seq_len, d_model)
    raise NotImplementedError("MultiHeadAttention forward pass not implemented.")

  def split(self, x):
    raise NotImplementedError("MultiHeadAttention split not implemented.")

  def concat(self, x):
    raise NotImplementedError("MultiHeadAttention concat not implemented.")



def scaled_dot_product_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  mask: Optional[torch.Tensor] = None,
  dropout: float = 0.1
) -> torch.Tensor:
  """
  Computes scaled dot product attention.

  Args:
    query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k)
    key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k)
    value (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v); typically seq_len_v == seq_len_k
    mask (Optional[torch.Tensor]): A tensor broadcastable to shape (..., seq_len_q, seq_len_k) that prevents attention to certain positions
    dropout (float): Dropout probability to apply on attention weights

  Returns:
    output (torch.Tensor): Result of attention, of shape (..., seq_len_q, d_v)
    attn (torch.Tensor): Attention weights of shape (..., seq_len_q, seq_len_k)
  """
  d_k = key.size(-1) # dim of keys

  # if query shape = (..., seq_len, d_k) and key shape = (..., seq_len, d_k)
  # key.transpose(-2, -1) = (..., d_k, seq_len) and result shape = (..., seq_len, seq_len)
  scores = torch.matmul(query, key.transpose(-2, -1))  / math.sqrt(d_k)

  if mask is not None: # TODO: apply masking to scores
    pass

  attn = torch.softmax(scores, dim=1) # weights

  if dropout > 0: # TODO: apply dropout to attn
    pass

  # if attn shape (..., seq_len, seq_len) and value shape (..., seq_len, d_v),
  # then output shape (..., seq_len, d_v)
  output = torch.matmul(attn, value)

  return output, attn