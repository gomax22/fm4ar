"""
Unit tests for fm4ar.nn.resnets
"""

import pytest
import torch

from fm4ar.nn.resnets import DenseResidualNet, InitialLayerForZeroInputs


def test__initial_layer_for_zero_inputs() -> None:
    """
    Test InitialLayerForZeroInputs returns zeros with correct shape.
    """
    layer = InitialLayerForZeroInputs(output_dim=5)
    x = torch.zeros(17, 0)
    out = layer(x)
    assert out.shape == (17, 5)
    assert torch.equal(out, torch.zeros(17, 5))


def test__dense_residual_net() -> None:
    """
    Test DenseResidualNet with various configurations.
    """

    # ---------------------------
    # Case 1: Illegal input shape
    # ---------------------------
    with pytest.raises(ValueError) as value_error:
        DenseResidualNet(input_shape=(1, 2, 3), output_dim=5, hidden_dims=())
    assert "DenseResidualNet only supports 1D inputs!" in str(value_error.value)

    # ---------------------------
    # Case 2: input_dim != 0 with context
    # ---------------------------
    net = DenseResidualNet(
        input_shape=(3,),
        output_dim=5,
        hidden_dims=(7, 11, 13),
        activation="ReLU",
        dropout=0.13,
        use_batch_norm=True,
        use_layer_norm=False,
        first_context_features=17,
        second_context_features=9,
    )
    x = torch.randn(19, 3)
    first_context = torch.randn(19, 17)
    second_context = torch.randn(19, 9)
    out = net(x=x, first_context=first_context, second_context=second_context)
    assert out.shape == (19, 5)

    # ---------------------------
    # Case 3: input_dim == 0 with context
    # ---------------------------
    net = DenseResidualNet(
        input_shape=(0,),
        output_dim=5,
        hidden_dims=(7,),
        activation="ReLU",
        dropout=0.0,
        use_batch_norm=False,
        use_layer_norm=True,
        first_context_features=17,
        second_context_features=None,
    )
    x = torch.randn(19, 0)
    first_context = torch.randn(19, 17)
    out = net(x=x, first_context=first_context)
    assert out.shape == (19, 5)

    # ---------------------------
    # Case 4: no context features, with final activation
    # ---------------------------
    net = DenseResidualNet(
        input_shape=(3,),
        output_dim=5,
        hidden_dims=(7,),
        final_activation="Sigmoid",
    )
    x = torch.randn(19, 3)
    out = net(x=x)
    assert out.shape == (19, 5)
    # Values should be in range [0,1] due to sigmoid
    assert torch.all(out >= 0) and torch.all(out <= 1)
