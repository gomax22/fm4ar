"""
Unit tests for patch_embeds.py
"""

import pytest
import torch
from torch import nn
from fm4ar.nn.patch_embeds import PatchEmbedding


# ============================================================
# Initialization Tests
# ============================================================

def test_init_valid():
    """Ensure valid initialization works."""
    patch_embed = PatchEmbedding(in_channels=3, patch_size=4, emb_dim=8, seq_length=16)
    assert isinstance(patch_embed.proj, nn.Conv1d)
    assert patch_embed.num_patches == 4
    assert patch_embed.proj.kernel_size == (4,)
    assert patch_embed.proj.stride == (4,)


def test_init_invalid_seq_length():
    """Test assertion when seq_length is not divisible by patch_size."""
    with pytest.raises(AssertionError) as e:
        _ = PatchEmbedding(in_channels=3, patch_size=5, emb_dim=8, seq_length=16)
    assert "Signal length must be divisible by patch size." in str(e.value)


# ============================================================
# Forward Pass Tests
# ============================================================

@pytest.mark.parametrize("batch_size,in_channels,patch_size,emb_dim,seq_length", [
    (2, 3, 4, 8, 16),
    (1, 1, 2, 4, 8),
    (4, 2, 8, 16, 32),
])
def test_forward_output_shape(batch_size, in_channels, patch_size, emb_dim, seq_length):
    """Check that the output shape is as expected."""
    model = PatchEmbedding(in_channels, patch_size, emb_dim, seq_length)
    x = torch.randn(batch_size, in_channels, seq_length)
    y = model(x)

    expected_num_patches = seq_length // patch_size
    assert y.shape == (batch_size, expected_num_patches, emb_dim)
    assert isinstance(y, torch.Tensor)


def test_forward_value_consistency():
    """Check that projection + transpose produce correct results."""
    model = PatchEmbedding(in_channels=1, patch_size=2, emb_dim=2, seq_length=4)
    with torch.no_grad():
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [B=1, C=1, L=4]
        y = model(x)
        # Output shape: [1, num_patches=2, emb_dim=2]
        assert y.shape == (1, 2, 2)
        # Sanity: verify linear transform consistency
        manual = model.proj(x).transpose(1, 2)
        torch.testing.assert_close(y, manual)


def test_forward_gradients():
    """Ensure gradients propagate through the convolution."""
    model = PatchEmbedding(in_channels=3, patch_size=4, emb_dim=8, seq_length=16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    y = model(x)
    y.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_forward_on_different_devices():
    """Ensure PatchEmbedding works on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    model = PatchEmbedding(in_channels=2, patch_size=4, emb_dim=8, seq_length=16).to(device)
    x = torch.randn(2, 2, 16, device=device)
    y = model(x)
    assert y.is_cuda
    assert y.shape == (2, 4, 8)


def test_repr_and_attributes():
    """Ensure model has expected attributes and string representation."""
    model = PatchEmbedding(1, 2, 4, 8)
    assert "Conv1d" in str(model)
    assert hasattr(model, "proj")
    assert hasattr(model, "num_patches")
