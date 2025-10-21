import pytest
import numpy as np
import torch
from unittest.mock import patch

from fm4ar.datasets.auxiliary_data_scalers import (
    IdentityScaler,
    MeanStdScaler,
    MinMaxScaler,
    get_auxiliary_data_scaler,
)


def test_identity_scaler() -> None:
    scaler = IdentityScaler()
    x = {"aux_data": np.array([1.0, 2.0, 3.0])}

    # Forward
    out = scaler.forward(x)
    assert np.array_equal(out["aux_data"], x["aux_data"])

    # Inverse
    out_inv = scaler.inverse(x)
    assert np.array_equal(out_inv["aux_data"], x["aux_data"])

    # Array and tensor helpers
    arr = np.array([4.0, 5.0])
    assert np.array_equal(scaler.forward_array(arr), arr)
    t = torch.tensor([6.0, 7.0])
    assert torch.allclose(scaler.forward_tensor(t), t)


def test_mean_std_scaler() -> None:
    mean = np.array([1.0, 2.0])
    std = np.array([2.0, 4.0])
    scaler = MeanStdScaler(mean=mean, std=std)

    x = {"aux_data": np.array([[3.0, 6.0]])}
    expected_forward = (x["aux_data"] - mean) / std
    expected_inverse = expected_forward * std + mean

    # Forward
    out = scaler.forward(x)
    assert np.allclose(out["aux_data"], expected_forward)

    # Inverse
    out_inv = scaler.inverse(out)
    assert np.allclose(out_inv["aux_data"], x["aux_data"])


def test_min_max_scaler() -> None:
    minimum = np.array([1.0, 2.0])
    maximum = np.array([5.0, 10.0])
    scaler = MinMaxScaler(minimum=minimum, maximum=maximum)

    x = {"aux_data": np.array([[3.0, 6.0]])}
    expected_forward = (x["aux_data"] - minimum) / (maximum - minimum)
    expected_inverse = expected_forward * (maximum - minimum) + minimum

    # Forward
    out = scaler.forward(x)
    assert np.allclose(out["aux_data"], expected_forward)

    # Inverse
    out_inv = scaler.inverse(out)
    assert np.allclose(out_inv["aux_data"], x["aux_data"])


@patch("fm4ar.nn.aux_data_scalers.load_normalization_params")
def test_get_auxiliary_data_scaler_identity(mock_load):
    config = {"method": "identity"}
    scaler = get_auxiliary_data_scaler(config)
    assert isinstance(scaler, IdentityScaler)


@patch("fm4ar.nn.aux_data_scalers.load_normalization_params")
def test_get_auxiliary_data_scaler_mean_std(mock_load):
    mean = np.array([0.0, 1.0])
    std = np.array([1.0, 2.0])
    mock_load.return_value = {
        "aux_data": {"train": {"mean": mean, "std": std}}
    }

    config = {"method": "mean_std", "kwargs": {"dataset": "inaf"}}
    scaler = get_auxiliary_data_scaler(config)
    assert isinstance(scaler, MeanStdScaler)
    assert np.array_equal(scaler.mean, mean)
    assert np.array_equal(scaler.std, std)


@patch("fm4ar.nn.aux_data_scalers.load_normalization_params")
def test_get_auxiliary_data_scaler_min_max(mock_load):
    minimum = np.array([0.0, 1.0])
    maximum = np.array([10.0, 5.0])
    mock_load.return_value = {
        "aux_data": {"train": {"min": minimum, "max": maximum}}
    }

    config = {"method": "min_max", "kwargs": {"dataset": "inaf"}}
    scaler = get_auxiliary_data_scaler(config)
    assert isinstance(scaler, MinMaxScaler)
    assert np.array_equal(scaler.minimum, minimum)
    assert np.array_equal(scaler.maximum, maximum)


def test_forward_inverse_tensor_consistency():
    mean = np.array([1.0, 2.0])
    std = np.array([2.0, 4.0])
    scaler = MeanStdScaler(mean=mean, std=std)
    x_tensor = torch.tensor([[3.0, 6.0]])
    y_tensor = scaler.forward_tensor(x_tensor)
    x_recovered = scaler.inverse_tensor(y_tensor)
    assert torch.allclose(x_tensor, x_recovered)
