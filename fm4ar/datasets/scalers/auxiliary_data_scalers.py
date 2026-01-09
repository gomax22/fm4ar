"""
Methods for scaling the target parameters `aux_data`.

Note: The `aux_data` scaling is different from the `data_transforms`, as
the it remains fixed for the entire training process while the data
transforms can change between different stages of training.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from fm4ar.utils.paths import expand_env_variables_in_path

class AuxiliaryDataScaler(ABC):
    """
    Base class for all aux_data scalers.
    """

    @abstractmethod
    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        return self.forward({"aux_data": x})["aux_data"]

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
        return self.inverse({"aux_data": x})["aux_data"]

    def inverse_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.from_numpy(self.inverse_array(x.cpu().numpy()))
            .type_as(x)
            .to(x.device)
        )


class IdentityScaler(AuxiliaryDataScaler):
    """
    Identity scaler for `aux_data`.
    """

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)


class MeanStdScaler(AuxiliaryDataScaler):
    """
    Scale `aux_data` by subtracting the mean and dividing by the std. dev.
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
        output["aux_data"] = (x["aux_data"] - self.mean) / self.std
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["aux_data"] = x["aux_data"] * self.std + self.mean
        return output


class MinMaxScaler(AuxiliaryDataScaler):
    """
    Scale `aux_data` by mapping it into the interval [0, 1].
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
        output["aux_data"] = (x["aux_data"] - self.minimum) / self.difference
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["aux_data"] = x["aux_data"] * self.difference + self.minimum
        return output


def get_auxiliary_data_scaler(config: dict[str, Any]) -> AuxiliaryDataScaler:
    """
    Get the scaler for auxiliary data specified in the given `config` (which is
    only the config for the feature scaler for auxiliary data, not the entire
    experiment config).
    """

    # Case 1: No feature scaling defined for theta
    if not config:
        return IdentityScaler()

    # Case 2: Feature scaling defined
    scaler: AuxiliaryDataScaler
    method = config["method"]
    kwargs = config.get("kwargs", {})
    match method:
        case "mean_std" | "MeanStdScaler":
            mean, std = get_mean_and_std(**kwargs)
            scaler = MeanStdScaler(mean=mean, std=std)
        case "min_max" | "MinMaxScaler":
            minimum, maximum = get_min_and_max(**kwargs)
            scaler = MinMaxScaler(minimum=minimum, maximum=maximum)
        case "identity" | "IdentityScaler":
            scaler = IdentityScaler()
        case _:
            raise ValueError(f"Unknown feature scaling method: {method}")

    return scaler


def get_mean_and_std(dataset: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the mean and standard deviation of the auxiliary data parameters.
    """

    match dataset:
        case "inaf":
            from fm4ar.datasets.inaf import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            mean = norm_params['aux_data']['train']['mean']
            std = norm_params['aux_data']['train']['std']
            del norm_params
        case "inara_subset":
            from fm4ar.datasets.inara_subset import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            mean = norm_params['aux_data']['train']['mean']
            std = norm_params['aux_data']['train']['std']
            del norm_params
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
    return mean, std


def get_min_and_max(dataset: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum of the auxiliary data parameters.
    """

    match dataset:
        case "inaf":
            from fm4ar.datasets.inaf import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            minimum = norm_params['aux_data']['train']['min']
            maximum = norm_params['aux_data']['train']['max']
            del norm_params
        case "inara_subset":
            from fm4ar.datasets.inara_subset import load_normalization_params
            norm_params = load_normalization_params(
                file_path=expand_env_variables_in_path(
                    kwargs.get("file_path", None)
                )
            )
            minimum = norm_params['aux_data']['train']['min']
            maximum = norm_params['aux_data']['train']['max']
            del norm_params
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")

    return minimum, maximum
