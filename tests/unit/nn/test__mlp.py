import pytest
import torch
import torch.nn as nn

from fm4ar.nn.mlp import MLP, MLPBlock


def test__mlp() -> None:
    """
    Test `fm4ar.nn.mlp.MLP`.
    """

    # Case 1: Single hidden layer, no dropout or batch norm
    mlp = MLP(
        input_dim=10,
        hidden_dims=[5],
        output_dim=1,
        activation="Tanh",
        dropout=0.0,
        layer_norm=True,
    )
    assert isinstance(mlp, nn.Module)
    assert isinstance(mlp.mlp, nn.Sequential)
    # Linear + Tanh + LayerNorm + Linear
    assert len(mlp.mlp) == 4
    assert isinstance(mlp.mlp[0], nn.Linear)
    assert isinstance(mlp.mlp[1], nn.Tanh)
    assert isinstance(mlp.mlp[2], nn.LayerNorm)
    assert isinstance(mlp.mlp[3], nn.Linear)
    assert mlp(torch.randn(7, 10)).shape == (7, 1)

    # Case 2: Multiple hidden layers with dropout and batch norm
    mlp = MLP(
        input_dim=10,
        hidden_dims=[5, 5],
        output_dim=1,
        activation="SiLU",
        dropout=0.5,
        batch_norm=True,
    )
    assert isinstance(mlp, nn.Module)
    assert isinstance(mlp.mlp, nn.Sequential)
    # Linear + SiLU + BatchNorm + Dropout + Linear + SiLU + BatchNorm + Dropout + Linear
    assert len(mlp.mlp) == 9
    assert isinstance(mlp.mlp[0], nn.Linear)
    assert isinstance(mlp.mlp[1], nn.SiLU)
    assert isinstance(mlp.mlp[2], nn.BatchNorm1d)
    assert isinstance(mlp.mlp[3], nn.Dropout)
    assert isinstance(mlp.mlp[4], nn.Linear)
    assert isinstance(mlp.mlp[5], nn.SiLU)
    assert isinstance(mlp.mlp[6], nn.BatchNorm1d)
    assert isinstance(mlp.mlp[7], nn.Dropout)
    assert isinstance(mlp.mlp[8], nn.Linear)
    assert mlp(torch.randn(7, 10)).shape == (7, 1)

    # Case 3: Both batch and layer norm should raise ValueError
    with pytest.raises(ValueError) as value_error:
        MLP(
            input_dim=10,
            hidden_dims=[5],
            output_dim=1,
            activation="ReLU",
            batch_norm=True,
            layer_norm=True,
        )
    assert "Can't use both batch and layer" in str(value_error.value)


def test__mlpblock() -> None:
    """
    Test `fm4ar.nn.mlp.MLPBlock`.
    """

    # Case 1: Standard MLPBlock
    block = MLPBlock(hidden_size=8, mlp_dim=16, dropout_rate=0.1, act="ReLU")
    x = torch.randn(4, 8)
    y = block(x)
    assert y.shape == x.shape
    # Ensure output is on the same device as input
    assert y.device == x.device

    # Case 2: MLPBlock with mlp_dim=0 should use hidden_size
    block = MLPBlock(hidden_size=6, mlp_dim=0, dropout_rate=0.0, act="SiLU")
    x = torch.randn(3, 6)
    y = block(x)
    assert y.shape == x.shape

    # Case 3: MLPBlock with GEGLU activation doubles linear1 output
    block = MLPBlock(hidden_size=5, mlp_dim=10, dropout_rate=0.0, act="GEGLU")
    x = torch.randn(2, 5)
    y = block(x)
    assert y.shape == x.shape

    # Case 4: Invalid dropout_rate
    with pytest.raises(ValueError) as excinfo:
        MLPBlock(hidden_size=5, mlp_dim=10, dropout_rate=1.5)
    assert "dropout_rate should be between 0 and 1" in str(excinfo.value)
