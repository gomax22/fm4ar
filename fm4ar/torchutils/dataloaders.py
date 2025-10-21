"""
Utilities for building PyTorch `DataLoader` objects.
"""

import platform
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from fm4ar.utils.multiproc import get_number_of_available_cores


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    n_workers: int,
    drop_last: bool,
    pin_memory: bool = True,
    random_seed: int = 42,
) -> DataLoader:
    """
    Build a `DataLoader` for the given `dataset`.

    Args:
        dataset: Dataset to build the `DataLoader` for.
        batch_size: Batch size for the `DataLoader`.
        shuffle: Whether to shuffle the data at every epoch.
        n_workers: Number of workers for the `DataLoader`.
        drop_last: Whether to drop the last batch if it is smaller than
            `batch_size`.
        pin_memory: Whether to use pinned memory for the `DataLoader`.
        random_seed: Random seed for reproducibility.

    Returns:
        The `DataLoader`.
    """

    # Build the test loader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=n_workers,
        persistent_workers=(n_workers > 0),
        generator=torch.Generator().manual_seed(random_seed),
    )

    return dataloader

def build_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    batch_size: int,
    n_workers: int,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation `DataLoaders` for the given `dataset`.

    Args:
        train_dataset: Dataset to use for training.
        valid_dataset: Dataset to use for validation.
        n_train_samples: Number of samples to use for training.
        n_valid_samples: Number of samples to use for validation.
        batch_size: Batch size for the train and test loaders.
        n_workers: Number of workers for the train and test loaders.
        drop_last: Whether to drop the last batch if it is smaller than
            `batch_size`. This is only used for the train loader.
        random_seed: Random seed for reproducibility.

    Returns:
        A 2-tuple: `(train_loader, valid_loader)`.
    """

    # Build the train loader
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        n_workers=n_workers,
        drop_last=False,
        pin_memory=True,
        random_seed=random_seed + 1,
    )

    # Build the validation loader
    valid_loader = build_dataloader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        n_workers=n_workers,
        drop_last=False,
        pin_memory=True,
        random_seed=random_seed + 2,
    )

    return train_loader, valid_loader


def get_number_of_workers(n_workers: int | Literal["auto"] = "auto") -> int:
    """
    Determine the number of workers for a `DataLoader`.

    Args:
        n_workers: If an integer is given, this is returned. If "auto"
            is given, we determine the number of cores based on the
            host system. Any other value raises a `ValueError`.

    Returns:
        Number of workers for the `DataLoader`.
    """

    # If an explicit number of workers is given, return it
    if isinstance(n_workers, int):
        return n_workers

    # Otherwise, determine the number depending on the host system
    elif n_workers == "auto":

        # On a Mac, the number of workers needs to be 0
        if platform.system() == "Darwin":
            return 0

        # Otherwise, use all but one available cores (but at least one)
        n_available_cores = get_number_of_available_cores()
        return max(n_available_cores - 1, 1)

    # Otherwise, raise an error
    else:
        raise ValueError("Invalid value for `n_workers`!")
