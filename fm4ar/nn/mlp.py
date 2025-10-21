# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines a function to build a multi-layer perceptron (MLP) in PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence

from fm4ar.torchutils.general import get_activation_from_name


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str,
        batch_norm: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """
        Instantiate an MLP with the given parameters.

        Args:
            input_dim: Dimension of the input.
            hidden_dims: List of hidden dimensions.
            output_dim: Dimension of the output.
            activation: Name of the activation function.
            batch_norm: Whether to use batch normalization.
            layer_norm: Whether to use layer normalization.
            dropout: Dropout probability (between 0 and 1).

        Returns:
            A multi-layer perceptron with the given parameters.
        """

        super().__init__()

        # Prepare list of layers
        layers = torch.nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        # Sanity check: Can't use both batch and layer normalization
        if batch_norm and layer_norm:
            raise ValueError("Can't use both batch and layer normalization.")

        # Note: There seems to be no clear consensus about the order of the
        # activation function and the batch normalization layer.
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(get_activation_from_name(activation))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
                if layer_norm:
                    layers.append(torch.nn.LayerNorm(dims[i + 1]))
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        """

        return torch.Tensor(self.mlp(x))


class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: str = "GELU"
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim) if act != "GEGLU" else nn.Linear(hidden_size, mlp_dim * 2)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = get_activation_from_name(act)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x