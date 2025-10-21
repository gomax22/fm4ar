"""
Unit tests for `fm4ar.datasets.load_dataset`.
"""

import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import Dataset

from fm4ar.datasets import load_dataset


def test_load_dataset_vasist(monkeypatch):
    """Test loading the Vasist dataset."""

    # Create mock datasets
    mock_train = MagicMock(spec=Dataset)
    mock_valid = MagicMock(spec=Dataset)
    mock_test = MagicMock(spec=Dataset)

    # Mock VasistDatasetConfig and load_vasist_dataset
    mock_config_class = MagicMock()
    mock_config_instance = MagicMock()
    mock_config_class.return_value = mock_config_instance
    mock_load_dataset = MagicMock(return_value=(mock_train, mock_valid, mock_test))

    monkeypatch.setattr("fm4ar.datasets.load_dataset.VasistDatasetConfig", mock_config_class)
    monkeypatch.setattr("fm4ar.datasets.load_dataset.load_vasist_dataset", mock_load_dataset)
    monkeypatch.setattr("fm4ar.datasets.load_dataset.expand_env_variables_in_path", lambda x: x)

    # Config dict
    config = {
        "dataset": {
            "name": "vasist_2023",
            "file_path": "/fake/path",
            "n_train_samples": 10,
            "n_valid_samples": 5,
            "n_test_samples": 3,
        },
        "theta_scaler": {},
    }

    train, valid, test = load_dataset(config)

    # Assertions
    assert train is mock_train
    assert valid is mock_valid
    assert test is mock_test
    mock_load_dataset.assert_called_once()


def test_load_dataset_inaf(monkeypatch):
    """Test loading the INAF dataset."""

    # Create mock datasets
    mock_train = MagicMock(spec=Dataset)
    mock_valid = MagicMock(spec=Dataset)
    mock_test = MagicMock(spec=Dataset)

    # Mock INAFDatasetConfig and load_inaf_dataset
    mock_config_class = MagicMock()
    mock_config_instance = MagicMock()
    mock_config_class.return_value = mock_config_instance
    mock_load_dataset = MagicMock(return_value=(mock_train, mock_valid, mock_test))

    monkeypatch.setattr("fm4ar.datasets.load_dataset.INAFDatasetConfig", mock_config_class)
    monkeypatch.setattr("fm4ar.datasets.load_dataset.load_inaf_dataset", mock_load_dataset)
    monkeypatch.setattr("fm4ar.datasets.load_dataset.expand_env_variables_in_path", lambda x: x)
    monkeypatch.setattr("fm4ar.datasets.load_dataset.get_theta_scaler", lambda cfg: "theta_scaler")
    monkeypatch.setattr("fm4ar.datasets.load_dataset.get_auxiliary_data_scaler", lambda cfg: "aux_scaler")

    # Config dict
    config = {
        "dataset": {
            "name": "inaf",
            "data_dir": "/fake/data/dir",
            "limit": None,
            "verbose": True,
        },
        "theta_scaler": {},
        "auxiliary_data_scaler": {},
    }

    train, valid, test = load_dataset(config)

    # Assertions
    assert train is mock_train
    assert valid is mock_valid
    assert test is mock_test
    mock_load_dataset.assert_called_once()


def test_load_dataset_unknown_dataset():
    """Test that an unknown dataset name raises ValueError."""

    config = {
        "dataset": {
            "name": "unknown_dataset",
        }
    }

    with pytest.raises(ValueError) as excinfo:
        load_dataset(config)

    assert "Unknown dataset name" in str(excinfo.value)
