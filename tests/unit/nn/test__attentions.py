"""
Unit tests for attentions.py (SABlock).
"""

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from fm4ar.nn.attentions import SABlock


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dummy_input():
    """Returns a batch of dummy inputs."""
    batch_size = 2
    seq_len = 5
    hidden_size = 8
    x = torch.randn(batch_size, seq_len, hidden_size)
    return x, batch_size, seq_len, hidden_size


# ============================================================
# Initialization Tests
# ============================================================

def test_invalid_dropout_rate():
    with pytest.raises(ValueError):
        SABlock(hidden_size=16, num_heads=4, dropout_rate=-0.1)

    with pytest.raises(ValueError):
        SABlock(hidden_size=16, num_heads=4, dropout_rate=1.5)


def test_invalid_head_divisibility():
    with pytest.raises(ValueError):
        SABlock(hidden_size=15, num_heads=4)


def test_causal_requires_sequence_length():
    with pytest.raises(ValueError):
        SABlock(hidden_size=16, num_heads=4, causal=True)


def test_flash_attention_requires_xformers(monkeypatch):
    # Force xformers to appear unavailable
    monkeypatch.setattr("fm4ar.nn.attentions.has_xformers", False)
    with pytest.raises(ValueError):
        SABlock(hidden_size=16, num_heads=4, use_flash_attention=True)


# ============================================================
# Forward Pass Tests
# ============================================================

def test_forward_shape(dummy_input):
    x, b, t, c = dummy_input
    block = SABlock(hidden_size=c, num_heads=4)
    y = block(x)

    # Output shape should match input shape
    assert y.shape == (b, t, c)
    assert isinstance(y, torch.Tensor)


def test_forward_with_mask(dummy_input):
    x, b, t, c = dummy_input
    block = SABlock(hidden_size=c, num_heads=2)
    # create a 4D mask (b, 1, t, t)
    mask = torch.ones(b, 1, t, t)
    mask[:, :, 1:, 0] = 0  # mask out certain positions

    y = block(x, mask=mask)
    assert y.shape == (b, t, c)


def test_forward_with_cross_attention(dummy_input):
    x, b, t, c = dummy_input
    context = torch.randn(b, t + 2, c)
    block = SABlock(hidden_size=c, num_heads=2, with_cross_attention=True)
    y = block(x, context=context)

    assert y.shape == (b, t, c)


def test_forward_with_causal_mask(dummy_input):
    x, b, t, c = dummy_input
    block = SABlock(hidden_size=c, num_heads=2, causal=True, sequence_length=t)
    y = block(x)
    assert y.shape == (b, t, c)

    # check causal mask exists and shape is correct
    assert hasattr(block, "causal_mask")
    assert block.causal_mask.shape[-2:] == (t, t)
    # Ensure upper triangle is masked
    assert torch.all(block.causal_mask[0, 0].tril() == block.causal_mask[0, 0])


def test_forward_differentiable(dummy_input):
    x, b, t, c = dummy_input
    x.requires_grad_(True)
    block = SABlock(hidden_size=c, num_heads=2)
    y = block(x)
    y.mean().backward()

    # Ensure gradients flow
    assert x.grad is not None
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"


# ============================================================
# Edge / Corner Cases
# ============================================================

def test_forward_single_token(dummy_input):
    _, _, _, c = dummy_input
    x = torch.randn(1, 1, c)
    block = SABlock(hidden_size=c, num_heads=1)
    y = block(x)
    assert y.shape == (1, 1, c)


def test_forward_large_dropout(dummy_input):
    x, b, t, c = dummy_input
    block = SABlock(hidden_size=c, num_heads=2, dropout_rate=0.9)
    y = block(x)
    assert y.shape == (b, t, c)


def test_mask_dimension_assertion(dummy_input):
    x, b, t, c = dummy_input
    block = SABlock(hidden_size=c, num_heads=2)
    bad_mask = torch.ones(b, t, t)  # should be 4D
    with pytest.raises(AssertionError):
        _ = block(x, mask=bad_mask)
