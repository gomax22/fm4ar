"""
Unit tests for `fm4ar.torchutils.general`.
"""

from types import SimpleNamespace
from typing import Type

import pytest
import torch

from fm4ar.nn.modules import Sine
from fm4ar.torchutils.general import (
    check_for_nans,
    move_batch_to_device,
    get_activation_from_name,
    get_cuda_info,
    get_number_of_parameters,
    resolve_device,
    set_random_seed,
)


def test__check_for_nans() -> None:
    """
    Test `fm4ar.torchutils.general.check_for_nans()`.
    """

    # Case 1: No NaN in the tensor
    tensor = torch.tensor([1.0, 2.0, 3.0])
    check_for_nans(tensor)

    # Case 2: NaN in the tensor
    tensor = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError) as value_error:
        check_for_nans(tensor, "my_tensor")
    assert "NaN values detected in my_tensor" in str(value_error)

    # Case 3: inf in the tensor
    tensor = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError) as value_error:
        check_for_nans(tensor, "my_tensor")
    assert "Inf values detected in my_tensor" in str(value_error)

    # Case 4: -inf in the tensor
    tensor = torch.tensor([1.0, float("-inf"), 3.0])
    with pytest.raises(ValueError) as value_error:
        check_for_nans(tensor, "my_tensor")
    assert "Inf values detected in my_tensor" in str(value_error)

def test_move_batch_to_device_cpu():
    """Test moving a batch to CPU and separating theta, context, and aux_data."""
    # Create a mock batch
    batch = {
        "theta": torch.randn(4, 3),
        "flux": torch.randn(4, 10),
        "wlen": torch.randn(4, 10),
        "aux_data": torch.randn(4, 2),
    }

    device = torch.device("cpu")
    theta, context, aux_data = move_batch_to_device(batch.copy(), device)

    # Check devices
    assert theta.device == device
    assert aux_data.device == device
    for val in context.values():
        assert val.device == device

    # Check separation
    assert "theta" not in context
    assert "aux_data" not in context
    assert set(context.keys()) == {"flux", "wlen"}
    assert theta.shape == (4, 3)
    assert aux_data.shape == (4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_batch_to_device_cuda():
    """Test moving a batch to GPU if available."""
    batch = {
        "theta": torch.randn(2, 5),
        "flux": torch.randn(2, 8),
        "wlen": torch.randn(2, 8),
    }

    device = torch.device("cuda")
    theta, context, aux_data = move_batch_to_device(batch.copy(), device)

    # Check devices
    assert theta.device == device
    for val in context.values():
        assert val.device == device

    # aux_data should be None if not present
    assert aux_data is None

    # Check separation
    assert "theta" not in context

@pytest.mark.parametrize(
    "activation_name, expected_activation",
    [
        ("ELU", torch.nn.ELU),
        ("GELU", torch.nn.GELU),
        ("LeakyReLU", torch.nn.LeakyReLU),
        ("ReLU", torch.nn.ReLU),
        ("Sigmoid", torch.nn.Sigmoid),
        ("Sine", Sine),
        ("SiLU", torch.nn.SiLU),  # same as Swish
        ("Tanh", torch.nn.Tanh),
        ("invalid", None),
    ],
)
def test__get_activation_from_string(
    activation_name: str,
    expected_activation: Type[torch.nn.Module],
) -> None:
    """
    Test `fm4ar.torchutils.general.get_activation_from_string()`.
    """

    if activation_name == "invalid":
        with pytest.raises(ValueError) as value_error:
            get_activation_from_name(activation_name)
        assert "Invalid activation function" in str(value_error)

    else:
        activation = get_activation_from_name(activation_name)
        assert isinstance(activation, expected_activation)


def test__get_cuda_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test `fm4ar.torchutils.general.get_cuda_info()`.
    """

    # Case 1: No CUDA devices available
    with monkeypatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: False)
        cuda_info = get_cuda_info()
        assert cuda_info == {}

    # Case 2: Pretend we have a CUDA device
    with monkeypatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: True)
        mp.setattr("torch.backends.cudnn.version", lambda: 123)
        mp.setattr("torch.version.cuda", "11.1")
        mp.setattr("torch.cuda.device_count", lambda: 1)
        mp.setattr("torch.cuda.get_device_name", lambda _: "GeForce GTX 1080")
        mp.setattr(
            "torch.cuda.get_device_properties",
            lambda _: SimpleNamespace(total_memory=8 * 1024 ** 3),
        )
        cuda_info = get_cuda_info()
        assert cuda_info == {
            "cuDNN version": 123,
            "CUDA version": "11.1",
            "device count": 1,
            "device name": "GeForce GTX 1080",
            "memory (GB)": 8.0,
        }


def test__get_number_of_parameters() -> None:
    """
    Test `fm4ar.torchutils.general.get_number_of_parameters()`.
    """

    layer_1 = torch.nn.Linear(10, 5)
    layer_1.requires_grad_(False)
    layer_2 = torch.nn.Linear(5, 1)
    model = torch.nn.Sequential(
        layer_1,
        torch.nn.ReLU(),
        layer_2,
    )

    n_trainable = get_number_of_parameters(model, (True,))
    n_fixed = get_number_of_parameters(model, (False,))
    n_total = get_number_of_parameters(model, (True, False))
    assert n_trainable == 6
    assert n_fixed == 55
    assert n_total == 61


def test__resolve_device() -> None:
    """
    Test `fm4ar.torchutils.general.resolve_device()`.
    """

    # Case 1: "cpu"
    device = resolve_device("cpu")
    assert device == torch.device("cpu")

    # Case 2: "cuda"
    device = resolve_device("cuda")
    assert device == torch.device("cuda")

    # Case 3: "auto" with cuda available
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: True)
        device = resolve_device("auto")
        assert device == torch.device("cuda")

    # Case 4: "auto" without cuda available
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: False)
        device = resolve_device("auto")
        assert device == torch.device("cpu")

    # Case 5: Invalid device
    with pytest.raises(RuntimeError) as runtime_error:
        resolve_device("invalid")
    assert "Expected one of" in str(runtime_error)


def test__set_random_seed(capsys: pytest.CaptureFixture):
    """
    Test `fm4ar.torchutils.general.set_random_seed()`.
    """

    # Case 1: Set random seed
    set_random_seed(seed=42, verbose=False)
    assert torch.initial_seed() == 42

    # Case 2: Set random seed again
    set_random_seed(seed=43, verbose=False)
    assert torch.initial_seed() == 43

    # Case 3: Set random seed again with verbose output
    set_random_seed(seed=423, verbose=True)
    assert torch.initial_seed() == 423
    captured = capsys.readouterr()
    assert "Set PyTorch random seed to 423" in captured.out
