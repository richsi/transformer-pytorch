import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
    """
    Args:
      d_model (int): Total dimension of the model
      num_heads (int): Number of attention heads
      dropout (float): Dropout probability on attention weights
    """
    super().__init__()
    assert d_model % num_heads == 0 # d_model must be divisible by num_heads

    self.d_model = d_model 
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.linear_q = nn.Linear(d_model, d_model)
    self.linear_k = nn.Linear(d_model, d_model)
    self.linear_v = nn.Linear(d_model, d_model)

    self.linear_out = nn.Linear(d_model, d_model)
    self.dropout = dropout

  def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      query (torch.Tensor): (batch_size, seq_len, d_model)
      key (torch.Tensor): (batch_size, seq_len, d_model)
      value (torch.Tensor): (batch_size, seq_len, d_model)
      mask (Optional[torch.Tensor]): (batch_size, 1, 1, seq_len) or broadcastable to that shape

    Returns:
      output (torch.Tensor): (batch_size, seq_len, d_model)
      attn (torch.Tensor): (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size = query.size(0)

    # 1. linear projection and splitting into multiple heads (batch_size, seq_len, d_model)
    Q = self.linear_q(query) # affine transformation
    K = self.linear_k(key)
    V = self.linear_v(value)

    # reshape to (batch_size, seq_len, num_heads, d_k) then tranpose to (batch_size, num_heads, seq_len, d_k)

    # 2. apply scaled dot product attention for each head in parallel
    # input (..., seq_len, d_k), output (.., seq_len, d_k)

    # 3. concatenate the heads together

    # 4. apply final linear projection



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