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
    self.d_head = d_model // num_heads

    self.W_q = nn.Linear(d_model, d_model) # transforms input to Q of all heads
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_o = nn.Linear(d_model, d_model) # output projection

    self.dropout = nn.Dropout(p=dropout)

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

    # linear projections (batch, seq_len, d_model) -> (batch_ seq_len, d_model)
    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)

    # split into n_heads and reshape for attention 
    Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2) # after split (batch, n_heads, seq_len, d_head)
    K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
    V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

    # merge batch and n_heads for efficiency: treat each head as a separate batch
    Q_flat = Q.reshape(batch_size * self.n_heads, -1, self.d_head)
    K_flat = K.reshape(batch_size * self.n_heads, -1, self.d_head)
    V_flat = V.reshape(batch_size * self.n_heads, -1, self.d_head)

    if mask is not None:
      mask = mask.repeat_interleave(self.n_heads, dim=0)
    attn_output, attn_weights = MultiHeadAttention.scaled_dot_product_attention(Q_flat, K_flat, V_flat, mask)
    # attn_output (batch * n_heads, seq_len_q, d_head)
    # reshape attn_output back to (batch_size, n_heads, seq_len, d_head)
    attn_output = attn_output.view(batch_size, self.n_heads, -1, self.d_head)
    # concatenate heads: (batch, seq_len, n_heads, d_head) reshape-> (batch, seq_len, n_heads * d_head)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)

    return self.W_o(attn_output) # (batch, seq_len, d_model)


  @staticmethod
  def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[torch.Tensor] = 0.1
  ) -> torch.Tensor:
    """
    Computes scaled dot product attention.

    Args:
      Q (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k)
      K (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k)
      V (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v); typically seq_len_v == seq_len_k
      mask (Optional[torch.Tensor]): A tensor broadcastable to shape (..., seq_len_q, seq_len_k) that prevents attention to certain positions
      dropout (Optional[torch.Tensor]): Dropout probability

    Returns:
      output (torch.Tensor): Result of attention, of shape (..., seq_len_q, d_v)
      attn (torch.Tensor): Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1) # dim of keys

    # if query shape = (..., seq_len, d_k) and key shape = (..., seq_len, d_k)
    # key.transpose(-2, -1) = (..., d_k, seq_len) and result shape = (..., seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None: 
      scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=1) # shape (batch, seq_len_q, seq_len_k)

    if dropout is not None:
      d = nn.Dropout(p=dropout)
      attn = d(attn)

    # if attn shape (..., seq_len, seq_len) and value shape (..., seq_len, d_v),
    # then output shape (..., seq_len, d_v)
    output = torch.matmul(attn, V)
    return output, attn