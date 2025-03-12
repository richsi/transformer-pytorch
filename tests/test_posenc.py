import torch
import pytest

from transformer.positional_encoding import SinusoidalPositionalEncoding 

@pytest.fixture
def pos_enc_cpu():
  max_len = 100
  d_model = 16
  device = "cpu"
  return SinusoidalPositionalEncoding(max_len, d_model, device)

def test_pe_shape(pos_enc_cpu):
  max_len = 100
  d_model = 16
  assert pos_enc_cpu.pe.shape == (max_len, d_model), "SPE encoding tensor has incorrect shape."

def test_requires_grad(pos_enc_cpu):
  assert not pos_enc_cpu.pe.requires_grad, "SPE encoding tensor should not require grad."

def test_forward_output_shape(pos_enc_cpu):
  batch_size = 2
  seq_len = 50
  d_model = 16
  dummy_input = torch.zeros(batch_size, seq_len, d_model)
  pe_slice = pos_enc_cpu(dummy_input)
  assert pe_slice.shape == (seq_len, d_model), "Forward output has incorrect shape."

def test_forward_values(pos_enc_cpu):
  batch_size = 2
  seq_len = 50
  d_model = 16
  dummy_input = torch.zeros(batch_size, seq_len, d_model)
  pe_slice = pos_enc_cpu(dummy_input)
  expected_slice = pos_enc_cpu.pe[:seq_len, :]
  assert torch.allclose(pe_slice, expected_slice), "Forward output does not match expected encoding slice."