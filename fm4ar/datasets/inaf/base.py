from functools import lru_cache

import os
import pickle
import pandas as pd
import numpy as np
import torch
from typing import List
from pydantic import Field

from torch.utils.data import Dataset
from fm4ar.datasets.data_transforms import DataTransform
from fm4ar.datasets.scalers.theta_scalers import ThetaScaler, IdentityScaler as ThetaIdentityScaler
from fm4ar.datasets.scalers.auxiliary_data_scalers import AuxiliaryDataScaler, IdentityScaler as AuxIdentityScaler
from fm4ar.datasets.scalers.flux_scalers import FluxScaler, IdentityScaler as FluxIdentityScaler
from fm4ar.datasets.scalers.error_bars_scalers import ErrorBarsScaler, IdentityScaler as ErrorBarsIdentityScaler
from fm4ar.datasets.scalers.wlen_scalers import WavelengthScaler, IdentityScaler as WavelengthIdentityScaler
from fm4ar.datasets import DatasetConfig

# constants for INAF dataset
RJUP = 69911000
MJUP = 1.898e27
RSOL = 696340000
MSOL = 1.989e30


LABELS = {
    'planet_temp': r"$\mathrm{T_p}$",
    'log_H2O': r"$\log \mathrm{H_2O}$", 
    'log_CO2': r"$\log \mathrm{CO_2}$", 
    'log_CO': r"$\log \mathrm{CO}$",
    'log_CH4': r"$\log \mathrm{CH_4}$", 
    'log_NH3': r"$\log \mathrm{NH_3}$"
}


def load_normalization_params(
    file_path: str,
) -> dict[str, np.ndarray]:
    """
    Load normalization parameters from an INAF dataset file.

    Args:
        file_path: Path to the INAF dataset file.

    Returns:
        A 2-tuple `(mean_flux, std_flux)`.
    """

    # Load the normalization parameters
    with open(file_path, "rb") as f:
        norm_params = pickle.load(f)

    return norm_params 

class INAFDatasetConfig(DatasetConfig):
    """
    Configuration for the INAFDataset.
    """
    data_dir: str = Field(
        ...,
        description="Path to the INAF dataset directory.",
    )
    predict_log_temperature: bool = Field(
        ...,
        description="Whether to predict log temperature.",
    )
    limit: int | None = Field(
        ...,
        description="Limit the number of samples to load from the dataset.",
    )
    verbose: bool = Field(
        ...,
        description="Whether to print verbose output during dataset loading.",
    )


class INAFDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 split: str, 
                 predict_log_temperature: bool = False,
                 limit: int = None, 
                 verbose: bool = True, 
                 theta_scaler: ThetaScaler | None = None,
                 auxiliary_data_scaler: AuxiliaryDataScaler | None = None,
                 flux_scaler: FluxScaler | None = None,
                 error_bars_scaler: ErrorBarsScaler | None = None,
                 wlen_scaler: WavelengthScaler | None = None) -> None:
        super().__init__()
        assert limit is None or isinstance(limit, int), "Limit must be an integer or None."
        assert split in ["train", "val", "test", None], "Split must be one of 'train', 'val', 'test', or None."
        

        self.data_dir = data_dir
        self.split = split
        self.limit = limit
        self.verbose = verbose
        self.predict_log_temperature = predict_log_temperature

        # Load file names for the specified split
        self.fnames, self.split_file = self._load_data_split(split=split, limit=limit)

        # Load auxiliary data file paths
        self.aux_data = os.path.join(data_dir, "AuxillaryTable.csv")
        self.theta = os.path.join(data_dir, 
                                  "FM_Parameter_Table_with_log_temperature.csv" if self.predict_log_temperature
                                  else "FM_Parameter_Table.csv")
        
        # List of transformations that will be applied in __getitem__()
        self.data_transforms: list[DataTransform] = []

        # Scaling transform for the parameters `theta` (e.g., minmax scaling)
        self.theta_scaler = (
            theta_scaler if theta_scaler is not None else ThetaIdentityScaler()
        )

        # Scaling transform for the auxiliary data `aux_data`
        self.auxiliary_data_scaler = (
            auxiliary_data_scaler if auxiliary_data_scaler is not None else AuxIdentityScaler()
        )

        # Scaling transform for the flux data `flux` (e.g., mean-std scaling)
        self.flux_scaler = (
            flux_scaler if flux_scaler is not None else FluxIdentityScaler()
        )

        # Scaling transform for the error_bars data `error_bars` (e.g., mean-std scaling)
        self.error_bars_scaler = (
            error_bars_scaler if error_bars_scaler is not None else ErrorBarsIdentityScaler()
        )

        # Scaling transform for the wavelength data `wlen` (e.g., fixed scaling)
        self.wlen_scaler = (
            wlen_scaler if wlen_scaler is not None else WavelengthIdentityScaler()
        )


    def __getitem__(self, ind):

        # Load the data from the HDF5 file
        model = pd.read_hdf(self.fnames[ind])

        # Prepare the sample dict
        # TODO: only include requested components
        sample = {
            "theta" : self.get_parameters(indices=np.array(ind)).copy(),
            "wlen" : model['instrument_wlgrid'][:].to_numpy().copy(),
            "aux_data" : self.get_aux_data(indices=np.array(ind)).copy(),
            "flux": model['instrument_spectrum'][:].to_numpy().copy().astype(np.float32),
            "error_bars":  model['instrument_noise'][:].to_numpy().copy().astype(np.float32)
        }

        # Apply the feature scaling for the flux
        sample = self.flux_scaler.forward(sample)

        # Apply the feature scaling for the error_bars
        sample = self.error_bars_scaler.forward(sample)
        
        # Apply the feature scaling for the wavelength
        sample = self.wlen_scaler.forward(sample)

        # First apply the data transforms (e.g., adding noise)
        for transform in self.data_transforms:
            sample = transform.forward(sample)

        # Apply the feature scaling for the parameters `theta`
        sample = self.theta_scaler.forward(sample)

        # Apply the feature scaling for the parameters `aux_data`
        sample = self.auxiliary_data_scaler.forward(sample)

        # Convert everything to PyTorch tensors.
        # This step is not a transform because it is non-optional and should
        # always be very the last step, so that all transforms can work with
        # numpy arrays and we only convert to tensors once at the very end.
        sample_as_tensors = {
            key: torch.from_numpy(val).float() for key, val in sample.items()
        }

        return sample_as_tensors    
    
    def __len__(self) -> int:
        return len(self.fnames)


    @property
    @lru_cache(maxsize=None)
    def dim_theta(self) -> int:
        """
        Return the number of parameters in the dataset, i.e., the
        dimensionality of `theta`.
        """

        # The dimensionality of the parameters should not be modified by any
        # of the transformations, so we do not need to apply them here.

        return self.get_parameters(indices=np.array(0)).shape[-1]

    @property
    @lru_cache(maxsize=None)
    def dim_context(self) -> int:
        """
        Return the number of wavelength bins in the spectra, i.e., the
        dimensionality of the context (`flux` and `wlen`).
        """

        # The dimensionality of the spectra can be modified by some of the
        # data transformations, such as subsampling or re-binning. Therefore,
        # we need to infer the number of bins dynamically.

        # Note: Even in these cases, the number of bins must be the same for
        # all spectra (at least within one training stage), otherwise we can't
        # construct batches in the DataLoader.
        return pd.read_hdf(self.fnames[0])['instrument_spectrum'][:].to_numpy().shape[-1]

    @property
    @lru_cache(maxsize=None)
    def dim_auxiliary_data(self) -> int:
        """
        Return the number of auxiliary data dimensions in the dataset, i.e.,
        the dimensionality of `aux_data`.
        """

        # The dimensionality of the auxiliary data can be modified by some of
        # the data transformations, such as subsampling or re-binning.
        # Therefore, we need to infer the number of bins dynamically.

        # Note: Even in these cases, the number of bins must be the same for
        # all spectra (at least within one training stage), otherwise we can't
        # construct batches in the DataLoader.
        return self.get_aux_data(indices=np.array(0)).shape[-1]

    
    def get_parameters(
        self,
        indices: np.ndarray = None, 
        skip_planet_index_col: bool = True
    ) -> np.ndarray:
        assert self.theta.endswith('.csv'), "File must be a .csv file"
        
        try:

            # Load parameters
            theta = pd.read_csv(self.theta)

        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        # Drop planet_ID column if specified
        theta = theta.drop(columns=['planet_ID']) if skip_planet_index_col else theta
        
        # Convert to numpy array
        theta = theta.to_numpy(dtype=np.float32)

        # Return requested indices
        if indices is None:
            if self.split_file is not None:
                return theta[self.split_file, :]
            else:
                return theta
        else:
            return theta[indices, :] 

    def get_parameters_labels(self) -> List[str]:
        assert self.theta.endswith('.csv'), "File must be a .csv file"
        
        try:
            theta = pd.read_csv(self.theta)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.theta} not found.")
        
        columns = theta.columns.tolist()
        columns.remove('planet_ID')
        labels = [v for k,v in LABELS.items() if k in columns]
        if self.predict_log_temperature:
            labels[0] = r"$\log \mathrm{T_p}$"
        return labels
            

    def get_parameters_dim(self) -> int:
        assert self.theta.endswith('.csv'), "File must be a .csv file"
        
        try:
            theta = pd.read_csv(self.theta)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.theta} not found.")
        
        theta = theta.drop(columns=['planet_ID'])
        return theta.shape[1]
        
    def get_aux_data_labels(self) -> List[str]:
        assert self.aux_data.endswith('.csv'), "File must be a .csv file"
        
        try:
            aux_data = pd.read_csv(self.aux_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        labels = aux_data.columns.tolist()
        labels.remove('planet_ID')
        return labels
        
    def get_aux_data_dim(self) -> int:
        assert self.aux_data.endswith('.csv'), "File must be a .csv file"
        
        try:
            aux_data = pd.read_csv(self.aux_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        aux_data = aux_data.drop(columns=['planet_ID'])
        return aux_data.shape[1]
    
    def get_aux_data(
        self,
        indices: np.ndarray = None,
        skip_planet_index_col: bool = True
    ) -> np.ndarray: 
        assert self.aux_data.endswith('.csv'), "File must be a .csv file"
        
        try:
            # Import unit conversion constants
            from fm4ar.datasets.inaf import RSOL, RJUP, MJUP, MSOL

            # Load auxiliary data
            aux_data = pd.read_csv(self.aux_data)

        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        # Convert units
        aux_data['star_radius_m'] = aux_data['star_radius_m'] / RSOL
        aux_data['planet_radius_m'] = aux_data['planet_radius_m'] / RJUP
        aux_data['planet_mass_kg'] = aux_data['planet_mass_kg'] / MJUP
        aux_data['star_mass_kg'] = aux_data['star_mass_kg'] / MSOL

        # To be consistent with theta, take log10 of star_temperature when
        # predicting log temperature
        if self.predict_log_temperature:
            aux_data['star_temperature'] = np.log10(aux_data['star_temperature'])

        # Drop planet_ID column if specified
        aux_data = aux_data.drop(columns=['planet_ID']) if skip_planet_index_col else aux_data
        
        # Convert to numpy array
        aux_data = aux_data.to_numpy(dtype=np.float32)

        # Return requested indices
        if indices is None:
            if self.split_file is not None:
                return aux_data[self.split_file, :]
            else:
                return aux_data
        else:
            return aux_data[indices, :]


    def _load_data_split(self, split: str, limit: int = None) -> None:
        # Load samples file names from data directory
        fnames = np.array([
            os.path.join(self.data_dir, "h5s", f) 
            for f in os.listdir(os.path.join(self.data_dir, "h5s")) 
            if f.endswith('.h5')
        ]) 

        # Load split indices if split is specified
        try:
            split_file = np.load(
                os.path.join(
                    self.data_dir, 
                    f"{split}_indices.npy"
                )) if split is not None else None
        except FileNotFoundError:
            if self.verbose:  print(f"{split}_indices.npy not found in {self.data_dir}")
            split_file = None

        # Apply limit to split file if specified
        if limit is not None and split_file is not None:
            assert isinstance(limit, int) and  limit > 0, "Limit must be a positive integer for split file."
            if limit > len(split_file):
                if self.verbose: print(f"Limit {limit} exceeds available data size {len(split_file)} for split file. Using full split.")
            else:
                if self.verbose: print(f"Using first {limit} samples from {split} split for split file.")
                split_file = split_file[:limit]
        
        # Filter filenames based on split indices
        fnames = fnames[split_file].tolist() if split_file is not None else fnames.tolist()
        fnames = fnames[:limit] if limit is not None else fnames

        return fnames, split_file
    


# TODO: load train, val, test splits
def load_inaf_dataset(
        data_dir: str, 
        predict_log_temperature: bool = False,
        limit: int = None, 
        verbose: bool = True, 
        theta_scaler: ThetaScaler | None = None,
        auxiliary_data_scaler: AuxiliaryDataScaler | None = None,
        flux_scaler: FluxScaler | None = None,
        error_bars_scaler: ErrorBarsScaler | None = None,
        wlen_scaler: WavelengthScaler | None = None
    ) -> tuple[INAFDataset, INAFDataset, INAFDataset]:
    """
    Load an INAF dataset from the given data directory.
    """

    # Load the train dataset
    train_dataset = INAFDataset(
        data_dir=data_dir,
        split='train',
        predict_log_temperature=predict_log_temperature,
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
        flux_scaler=flux_scaler,
        error_bars_scaler=error_bars_scaler,
        wlen_scaler=wlen_scaler,
    )
    
    #  Load the validation dataset
    val_dataset = INAFDataset(
        data_dir=data_dir,
        predict_log_temperature=predict_log_temperature,
        split='val',
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
        flux_scaler=flux_scaler,
        error_bars_scaler=error_bars_scaler,
        wlen_scaler=wlen_scaler,
    )
    # Load the test dataset
    test_dataset = INAFDataset(
        data_dir=data_dir,
        predict_log_temperature=predict_log_temperature,
        split='test',
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
        flux_scaler=flux_scaler,
        error_bars_scaler=error_bars_scaler,
        wlen_scaler=wlen_scaler,
    )


    return train_dataset, val_dataset, test_dataset
    

