"""
Methods for scaling the target parameters `wlen`.

Note: The `wlen` scaling is different from the `data_transforms`, as
the it remains fixed for the entire training process while the data
transforms can change between different stages of training.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from fm4ar.utils.paths import expand_env_variables_in_path

class WavelengthScaler(ABC):
    """
    Base class for all wavelength scalers.
    """

    @abstractmethod
    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        return self.forward({"wlen": x})["wlen"]

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.from_numpy(self.forward_array(x.cpu().numpy()))
            .type_as(x)
            .to(x.device)
        )

    @abstractmethod
    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

    def inverse_array(self, x: np.ndarray) -> np.ndarray:
        return self.inverse({"wlen": x})["wlen"]

    def inverse_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.from_numpy(self.inverse_array(x.cpu().numpy()))
            .type_as(x)
            .to(x.device)
        )


class IdentityScaler(WavelengthScaler):
    """
    Identity scaler for `wlen`.
    """

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)


class UnitsScaler(WavelengthScaler):
    """
    Scale `wlen` using a given scale and offset.
    """

    def __init__(
        self,
        scale: np.ndarray,
        offset: np.ndarray,
    ) -> None:

        super().__init__()

        self.scale = scale
        self.offset = offset

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["wlen"] = (x["wlen"] * self.scale) + self.offset
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["wlen"] = (x["wlen"] - self.offset) / self.scale
        return output


def get_wlen_scaler(config: dict[str, Any]) -> WavelengthScaler:
    """
    Get the scaler for wavelength specified in the given `config` (which is
    only the config for the feature scaler for wlen, not the entire
    experiment config).
    """

    # Case 1: No feature scaling defined for wlen
    if not config:
        return IdentityScaler()

    # Case 2: Feature scaling defined
    scaler: WavelengthScaler
    method = config["method"]
    kwargs = config.get("kwargs", {})
    match method:
        case "units" | "UnitsScaler":
            scale = np.array(kwargs["scale"])
            offset = np.array(kwargs["offset"])
            scaler = UnitsScaler(scale=scale, offset=offset)
        case "identity" | "IdentityScaler":
            scaler = IdentityScaler()
        case _:
            raise ValueError(f"Unknown feature scaling method: {method}")

    return scaler

