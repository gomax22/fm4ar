import os
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from fm4ar.datasets.inaf import INAFDataset, load_inaf_dataset
from fm4ar.datasets.theta_scalers import IdentityScaler
from fm4ar.datasets.auxiliary_data_scalers import IdentityScaler as AuxIdentityScaler


@pytest.fixture
def mock_csv(tmp_path):
    # Create dummy CSVs for theta and aux_data
    theta_csv = tmp_path / "FM_Parameter_Table.csv"
    aux_csv = tmp_path / "AuxillaryTable.csv"

    theta_df = pd.DataFrame({
        "planet_ID": [0, 1],
        "param1": [1.0, 2.0],
        "param2": [3.0, 4.0],
    })
    theta_df.to_csv(theta_csv, index=False)

    aux_df = pd.DataFrame({
        "planet_ID": [0, 1],
        "star_radius_m": [1.0, 2.0],
        "planet_radius_m": [3.0, 4.0],
        "planet_mass_kg": [5.0, 6.0],
        "star_mass_kg": [7.0, 8.0],
    })
    aux_df.to_csv(aux_csv, index=False)

    # Create dummy HDF5 file
    h5_dir = tmp_path / "h5s"
    h5_dir.mkdir()
    h5_file = h5_dir / "dummy.h5"
    pd.DataFrame({
        "instrument_wlgrid": [np.array([1.0, 2.0])],
        "instrument_spectrum": [np.array([10.0, 20.0])],
        "instrument_noise": [np.array([0.1, 0.2])],
    }).to_hdf(h5_file, key='df', mode='w')

    # Create split index files
    np.save(tmp_path / "train_indices.npy", np.array([0]))
    np.save(tmp_path / "val_indices.npy", np.array([0]))
    np.save(tmp_path / "test_indices.npy", np.array([0]))

    return tmp_path


@patch("fm4ar.datasets.inaf.pd.read_hdf")
def test_getitem_and_len(mock_read_hdf, mock_csv):
    # Mock read_hdf to return a DataFrame similar to HDF5
    mock_df = pd.DataFrame({
        "instrument_wlgrid": [np.array([1.0, 2.0])],
        "instrument_spectrum": [np.array([10.0, 20.0])],
        "instrument_noise": [np.array([0.1, 0.2])]
    })
    mock_read_hdf.return_value = mock_df

    dataset = INAFDataset(
        data_dir=str(mock_csv),
        split="train",
        limit=None,
        theta_scaler=IdentityScaler(),
        auxiliary_data_scaler=AuxIdentityScaler(),
    )

    # __len__ returns number of files
    assert len(dataset) == 1

    sample = dataset[0]
    assert isinstance(sample, dict)
    for key in ["theta", "wlen", "flux", "error_bars", "aux_data"]:
        assert key in sample
        assert isinstance(sample[key], torch.Tensor)


def test_properties_and_aux_methods(mock_csv):
    dataset = INAFDataset(
        data_dir=str(mock_csv),
        split="train",
        limit=None,
        theta_scaler=IdentityScaler(),
        auxiliary_data_scaler=AuxIdentityScaler(),
    )

    # get_parameters
    params = dataset.get_parameters(np.array([0]))
    assert params.shape[1] == dataset.dim_theta

    # get_aux_data
    aux = dataset.get_aux_data(np.array([0]))
    assert aux.shape[1] == dataset.dim_auxiliary_data

    # Labels
    theta_labels = dataset.get_parameters_labels()
    aux_labels = dataset.get_aux_data_labels()
    assert all(isinstance(lbl, str) for lbl in theta_labels)
    assert all(isinstance(lbl, str) for lbl in aux_labels)

    # Dimensions
    assert dataset.get_parameters_dim() == dataset.dim_theta
    assert dataset.get_aux_data_dim() == dataset.dim_auxiliary_data


def test_load_inaf_dataset(mock_csv):
    train, val, test = load_inaf_dataset(data_dir=str(mock_csv))
    assert isinstance(train, INAFDataset)
    assert isinstance(val, INAFDataset)
    assert isinstance(test, INAFDataset)
    assert train.split == "train"
    assert val.split == "val"
    assert test.split == "test"


def test_invalid_split(mock_csv):
    with pytest.raises(AssertionError):
        INAFDataset(data_dir=str(mock_csv), split="invalid")
