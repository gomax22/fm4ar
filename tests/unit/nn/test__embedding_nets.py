"""
Tests for embedding_nets.py
"""

from copy import deepcopy

import pytest
import torch

from fm4ar.nn.embedding_nets import (
    TransformerEncoder,
    PositionalEncoding,
    SoftClipFlux,
    create_embedding_net,
)


def test__create_embedding_net() -> None:
    """
    Test `create_embedding_net()`.
    """

    # Case 1: Check for supports_dict_input=True
    block_configs = [
        {"block_type": "PositionalEncoding", "kwargs": {"n_freqs": 3}}
    ]
    create_embedding_net(  # This should work
        input_shape=(123,),
        block_configs=block_configs,
        supports_dict_input=False,
    )
    with pytest.raises(ValueError) as value_error:
        create_embedding_net(  # This should fail
            input_shape=(123,),
            block_configs=block_configs,
            supports_dict_input=True,
        )
    assert "The first block must be a `SupportsDictInput`!" in str(value_error)

    # Case 2: Standard use case
    block_configs = [
        {"block_type": "SoftClipFlux", "kwargs": {"bound": 10.0}},
        {"block_type": "Concatenate", "kwargs": {"keys": ["flux", "wlen"]}},
        {
            "block_type": "DenseResidualNet",
            "kwargs": {
                "output_dim": 5,
                "hidden_dims": (10, 10, 10),
            },
        },
    ]
    embedding_net, output_dim = create_embedding_net(
        input_shape=(123,),
        block_configs=block_configs,
        supports_dict_input=True,
    )
    assert isinstance(embedding_net, torch.nn.Sequential)
    assert len(embedding_net) == 3
    assert output_dim == 5
    assert embedding_net[2].initial_layer.in_features == 246
    dummy_input = {"flux": torch.randn(17, 123), "wlen": torch.randn(17, 123)}
    assert embedding_net(dummy_input).shape == (17, 5)


@pytest.mark.parametrize(
    "theta_dim, n_freqs, encode_theta",
    [
        (5, 1, True),
        (13, 3, False),
    ],
)
def test__positional_encoding(
    theta_dim: int,
    n_freqs: int,
    encode_theta: bool,
) -> None:
    """
    Test `PositionalEncoding`.
    """

    # Create a positional encoding module
    positional_encoding = PositionalEncoding(
        n_freqs=n_freqs,
        encode_theta=encode_theta,
    )

    # Create a batch with random input
    batch_size = 17
    t_theta = torch.randn(batch_size, 1 + theta_dim)
    encoded = positional_encoding(t_theta)

    # Check that the output has the correct shape
    assert encoded.shape == (
        batch_size,
        1 + theta_dim + 2 * (1 + int(encode_theta) * theta_dim) * n_freqs,
    )


def test__soft_clip_flux() -> None:
    """
    Test `SoftClipFlux`.
    """

    # Make sure that the original input is not modified
    x = dict(flux=100 * torch.randn(17, 123))
    x_orig = deepcopy(x)
    soft_clip_flux = SoftClipFlux(bound=10.0)
    soft_clip_flux(x)
    assert torch.equal(x["flux"], x_orig["flux"])

    # Check that the flux is clipped
    x = dict(flux=100 * torch.randn(17, 123))
    assert not torch.all(x["flux"] <= 10.0)
    assert not torch.all(x["flux"] >= -10.0)
    x_clipped = soft_clip_flux(x)
    assert torch.all(x_clipped["flux"] <= 10.0)
    assert torch.all(x_clipped["flux"] >= -10.0)


@pytest.mark.parametrize(
    "seq_length, patch_size, in_channels, attn_layers_dim, attn_layers_depth, attn_layers_heads",
    [
        (1024, 256, 1, 32, 1, 2),
        (2048, 512, 3, 64, 2, 4),
    ],
)  
def test__transformer_encoder(
    seq_length: int,
    patch_size: int,
    in_channels: int,
    attn_layers_dim: int,
    attn_layers_depth: int,
    attn_layers_heads: int,
) -> None:
    """
    Test `TransformerEncoder`.
    """

    # Create a TransformerEncoder
    transformer_encoder = TransformerEncoder(
        seq_length=seq_length,
        patch_size=patch_size,
        in_channels=in_channels,
        attn_layers_dim=attn_layers_dim,
        attn_layers_depth=attn_layers_depth,
        attn_layers_heads=attn_layers_heads,
        embedding_dropout_rate=0.1,
        use_flash_attention=False,
    )

    # Create a batch with random input
    batch_size = 8
    t_x = torch.randn(batch_size, in_channels, seq_length)
    encoded = transformer_encoder(t_x)
    # Check that the output has the correct shape
    n_patches = seq_length // patch_size
    assert encoded.shape == (batch_size, n_patches, 64)
