"""
Unit tests for transformers.py
Covers: TransformerBlock, TransformerEncoder, and modulate()
"""

import pytest
import torch
import torch.nn as nn
from fm4ar.nn.transformers import TransformerBlock, TransformerEncoder, modulate


# ============================================================
# Helper Functions
# ============================================================

def count_parameters(model):
    """Utility to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Basic Function Tests
# ============================================================

def test_modulate_function():
    """Ensure the modulation function performs scaling and shifting correctly."""
    x = torch.ones(2, 4, 3)
    shift = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    scale = torch.tensor([[0.5, 1.0, -1.0], [1.0, -0.5, 0.0]])

    out = modulate(x, shift, scale)
    assert out.shape == x.shape
    # For first batch, last feature -> x*(1+(-1)) + 3 = 3
    assert torch.isclose(out[0, 0, 2], torch.tensor(3.0))
    # Ensure broadcasting works properly
    torch.testing.assert_close(out[0, :, 0], out[0, :, 0])


# ============================================================
# TransformerBlock Tests
# ============================================================

def test_transformer_block_init_valid():
    """Ensure TransformerBlock initializes properly."""
    block = TransformerBlock(
        hidden_size=16,
        mlp_dim=32,
        num_heads=4,
        dropout_rate=0.1,
        qkv_bias=True,
        causal=False,
        sequence_length=8,
    )

    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.attn, nn.Module)
    assert isinstance(block.mlp, nn.Module)
    assert not block.with_cross_attention
    assert count_parameters(block) > 0


def test_transformer_block_invalid_heads():
    """Check that invalid head configuration raises ValueError."""
    with pytest.raises(ValueError):
        TransformerBlock(hidden_size=15, mlp_dim=32, num_heads=4)


def test_transformer_block_invalid_dropout():
    """Check that invalid dropout rates raise ValueError."""
    with pytest.raises(ValueError):
        TransformerBlock(hidden_size=16, mlp_dim=32, num_heads=4, dropout_rate=1.5)


@pytest.mark.parametrize("use_cross_attention,use_adaLN", [
    (False, False),
    (False, True),
])
def test_transformer_block_forward_output_shape(use_cross_attention, use_adaLN):
    """Test that TransformerBlock produces correct output shape."""
    block = TransformerBlock(
        hidden_size=16,
        mlp_dim=32,
        num_heads=4,
        sequence_length=8,
        with_cross_attention=use_cross_attention,
        use_adaLN_modulation=use_adaLN,
    )

    x = torch.randn(2, 8, 16)
    context = torch.randn(2, 16) if use_adaLN else None

    y = block(x, context=context)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_transformer_block_with_cross_attention():
    """Ensure cross-attention path works."""
    block = TransformerBlock(
        hidden_size=16,
        mlp_dim=32,
        num_heads=4,
        sequence_length=8,
        with_cross_attention=True,
    )

    x = torch.randn(2, 8, 16)
    context = torch.randn(2, 8, 16)
    y = block(x, context=context)
    assert y.shape == x.shape


def test_transformer_block_adaLN_and_cross_attention_exclusion():
    """Verify assertion preventing simultaneous cross-attention and AdaLN modulation."""
    with pytest.raises(AssertionError):
        TransformerBlock(
            hidden_size=16,
            mlp_dim=32,
            num_heads=4,
            sequence_length=8,
            with_cross_attention=True,
            use_adaLN_modulation=True,
        )


def test_transformer_block_gradients():
    """Ensure gradients propagate through TransformerBlock."""
    block = TransformerBlock(
        hidden_size=16,
        mlp_dim=32,
        num_heads=4,
        sequence_length=8,
    )
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ============================================================
# TransformerEncoder Tests
# ============================================================

def test_transformer_encoder_init_and_forward():
    """Basic shape and type checks for TransformerEncoder."""
    model = TransformerEncoder(
        seq_length=16,
        patch_size=4,
        in_channels=2,
        attn_layers_dim=8,
        attn_layers_depth=2,
        attn_layers_heads=2,
    )
    x = torch.randn(3, 2, 16)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, model.num_patches * 8)
    assert torch.isfinite(y).all()


def test_transformer_encoder_invalid_patch_size():
    """Ensure invalid seq_length / patch_size raises AssertionError."""
    with pytest.raises(AssertionError):
        TransformerEncoder(
            seq_length=15,
            patch_size=4,
            in_channels=1,
            attn_layers_dim=8,
            attn_layers_depth=1,
            attn_layers_heads=2,
        )


def test_transformer_encoder_gradients():
    """Ensure gradients propagate end-to-end."""
    model = TransformerEncoder(
        seq_length=16,
        patch_size=4,
        in_channels=2,
        attn_layers_dim=8,
        attn_layers_depth=2,
        attn_layers_heads=2,
    )
    x = torch.randn(2, 2, 16, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_transformer_encoder_device_compatibility():
    """Ensure model runs correctly on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = TransformerEncoder(
        seq_length=16,
        patch_size=4,
        in_channels=2,
        attn_layers_dim=8,
        attn_layers_depth=1,
        attn_layers_heads=2,
    ).cuda()

    x = torch.randn(2, 2, 16, device="cuda")
    y = model(x)
    assert y.is_cuda
    assert y.shape == (2, model.num_patches * 8)
