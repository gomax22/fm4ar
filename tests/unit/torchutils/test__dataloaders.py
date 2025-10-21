"""
Unit tests for `fm4ar.torchutils.dataloaders`.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import platform

from fm4ar.torchutils.dataloaders import (
    build_dataloader,
    build_dataloaders,
    get_number_of_workers,
)


@pytest.fixture
def dummy_dataset() -> TensorDataset:
    """
    Simple dummy dataset for testing.
    """
    x = torch.arange(10).float().unsqueeze(1)
    y = (torch.arange(10) % 2).float().unsqueeze(1)
    return TensorDataset(x, y)


# ------------------------------------------------------------------------------
# build_dataloader
# ------------------------------------------------------------------------------
def test_build_dataloader_returns_dataloader(dummy_dataset):
    """
    Check that `build_dataloader()` returns a valid DataLoader.
    """
    loader = build_dataloader(
        dataset=dummy_dataset,
        batch_size=2,
        shuffle=True,
        n_workers=0,
        drop_last=False,
        pin_memory=False,
        random_seed=123,
    )

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2
    assert loader.drop_last is False
    assert loader.num_workers == 0
    assert len(loader) == 5  # 10 samples / batch_size 2

    # Check data content consistency
    batches = list(loader)
    assert all(isinstance(b, (list, tuple)) for b in batches)
    assert all(len(b) == 2 for b in batches)  # (x, y) pairs
    x_batch, y_batch = batches[0]
    assert torch.allclose(x_batch, dummy_dataset.tensors[0][:2])


# ------------------------------------------------------------------------------
# build_dataloaders
# ------------------------------------------------------------------------------
def test_build_dataloaders_creates_two_loaders(dummy_dataset):
    """
    Test that `build_dataloaders()` returns two valid loaders.
    """
    train_loader, valid_loader = build_dataloaders(
        train_dataset=dummy_dataset,
        valid_dataset=dummy_dataset,
        batch_size=2,
        n_workers=0,
        random_seed=42,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(valid_loader, DataLoader)
    assert train_loader.batch_size == 2
    assert valid_loader.batch_size == 2

    # Check reproducibility: repeated call with same seed gives same first batch
    train_loader_2, _ = build_dataloaders(
        train_dataset=dummy_dataset,
        valid_dataset=dummy_dataset,
        batch_size=2,
        n_workers=0,
        random_seed=42,
    )

    batch1_x, _ = next(iter(train_loader))
    batch2_x, _ = next(iter(train_loader_2))
    assert torch.allclose(batch1_x, batch2_x)


# ------------------------------------------------------------------------------
# get_number_of_workers
# ------------------------------------------------------------------------------
def test_get_number_of_workers_explicit_int():
    """Test explicit integer value."""
    assert get_number_of_workers(8) == 8


def test_get_number_of_workers_auto_darwin(monkeypatch):
    """Test automatic worker detection on macOS."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert get_number_of_workers("auto") == 0


def test_get_number_of_workers_auto_linux(monkeypatch):
    """Test automatic worker detection on Linux."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    # Patch available cores to a fixed value
    import fm4ar.torchutils.dataloaders as dataloaders
    monkeypatch.setattr(dataloaders, "get_number_of_available_cores", lambda: 8)

    assert get_number_of_workers("auto") == 7  # 8 - 1
    monkeypatch.setattr(dataloaders, "get_number_of_available_cores", lambda: 1)
    assert get_number_of_workers("auto") == 1


def test_get_number_of_workers_invalid_value():
    """Test invalid input handling."""
    with pytest.raises(ValueError, match="Invalid value for `n_workers`!"):
        get_number_of_workers("invalid")  # type: ignore
