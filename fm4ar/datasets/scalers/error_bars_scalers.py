"""
Methods for scaling the target parameters `error_bars`.

Note: The `error_bars` scaling is different from the `data_transforms`, as
the it remains fixed for the entire training process while the data
transforms can change between different stages of training.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from fm4ar.utils.paths import expand_env_variables_in_path

class ErrorBarsScaler(ABC):
    """
    Base class for all error_bars scalers.
    """

    @abstractmethod
    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        return self.forward({"error_bars": x})["error_bars"]

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
        return self.inverse({"error_bars": x})["error_bars"]

    def inverse_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.from_numpy(self.inverse_array(x.cpu().numpy()))
            .type_as(x)
            .to(x.device)
        )


class IdentityScaler(ErrorBarsScaler):
    """
    Identity scaler for `error_bars`.
    """

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)


class MeanStdScaler(ErrorBarsScaler):
    """
    Scale `error_bars` by subtracting the mean and dividing by the std. dev.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:

        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["error_bars"] = (x["error_bars"] - self.mean) / self.std
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["error_bars"] = x["error_bars"] * self.std + self.mean
        return output


class MinMaxScaler(ErrorBarsScaler):
    """
    Scale `error_bars` by mapping it into the interval [0, 1].
    """

    def __init__(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> None:

        super().__init__()

        self.minimum = minimum
        self.maximum = maximum
        self.difference = self.maximum - self.minimum

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["error_bars"] = (x["error_bars"] - self.minimum) / self.difference
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["error_bars"] = x["error_bars"] * self.difference + self.minimum
        return output

class UnitsScaler(ErrorBarsScaler):
    """
    Scale `error_bars` using a given scale and offset.
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
        output["error_bars"] = (x["error_bars"] * self.scale) + self.offset
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["error_bars"] = (x["error_bars"] - self.offset) / self.scale
        return output

def get_error_bars_scaler(config: dict[str, Any]) -> ErrorBarsScaler:
    """
    Get the scaler for error_bars specified in the given `config` (which is
    only the config for the feature scaler for error_bars, not the entire
    experiment config).
    """

    # Case 1: No feature scaling defined for error_bars
    if not config:
        return IdentityScaler()

    # Case 2: Feature scaling defined
    scaler: ErrorBarsScaler
    method = config["method"]
    kwargs = config.get("kwargs", {})
    match method:
        case "mean_std" | "MeanStdScaler":
            mean, std = get_mean_and_std(**kwargs)
            scaler = MeanStdScaler(mean=mean, std=std)
        case "min_max" | "MinMaxScaler":
            minimum, maximum = get_min_and_max(**kwargs)
            scaler = MinMaxScaler(minimum=minimum, maximum=maximum)
        case "units" | "UnitsScaler":
            scale = np.array(kwargs["scale"])
            offset = np.array(kwargs["offset"])
            scaler = UnitsScaler(scale=scale, offset=offset)
        case "identity" | "IdentityScaler":
            scaler = IdentityScaler()
        case _:
            raise ValueError(f"Unknown feature scaling method: {method}")

    return scaler


def get_mean_and_std(dataset: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the mean and standard deviation of the error_bars
    """
    match dataset:
        case "inaf":
            from fm4ar.datasets.inaf import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            mean = norm_params["instrument_noise"]["train"]["mean"]
            std = norm_params["instrument_noise"]["train"]["std"]
            del norm_params
        case "inara_subset":
            from fm4ar.datasets.inara_subset import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            mean = norm_params["instrument_noise"]["train"]["mean"]
            std = norm_params["instrument_noise"]["train"]["std"]
            del norm_params
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    return mean, std


def get_min_and_max(dataset: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum of the error_bars parameters.
    """
    match dataset:
        case "inaf":
            from fm4ar.datasets.inaf import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            minimum = norm_params["instrument_noise"]["train"]["min"]
            maximum = norm_params["instrument_noise"]["train"]["max"]
            del norm_params
        case "inara_subset":
            from fm4ar.datasets.inara_subset import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            minimum = norm_params["instrument_noise"]["train"]["min"]
            maximum = norm_params["instrument_noise"]["train"]["max"]
            del norm_params

        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    return minimum, maximum
