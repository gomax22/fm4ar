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
from fm4ar.datasets.theta_scalers import IdentityScaler, ThetaScaler
from fm4ar.datasets.auxiliary_data_scalers import AuxiliaryDataScaler, IdentityScaler as AuxIdentityScaler
from fm4ar.datasets import DatasetConfig

# constants for INAF dataset
RJUP = 69911000
MJUP = 1.898e27
RSOL = 696340000
MSOL = 1.989e30


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
                 # components: List[str] = ['theta', 'flux', 'wlen', 'error_bars', 'aux_data'],
                 split: str = None, 
                 limit: int = None, 
                 verbose: bool = True, 
                 theta_scaler: ThetaScaler | None = None,
                 auxiliary_data_scaler: AuxiliaryDataScaler | None = None) -> None:
        super().__init__()
        assert limit is None or isinstance(limit, int), "Limit must be an integer or None."
        # assert 'theta' in components, "'theta' must be included in components."
        # assert 'flux' in components, "'flux' must be included in components."
        # assert 'wlen' in components, "'wlen' must be included in components."
        assert split in ["train", "val", "test", None], "Split must be one of 'train', 'val', 'test', or None."
        

        self.data_dir = data_dir
        # self.components = components
        self.split = split
        self.limit = limit
        self.verbose = verbose

        # Load file names for the specified split
        self.fnames, self.split_file = self._load_data_split(split=split, limit=limit)

        # Load auxiliary data file paths
        self.aux_data = os.path.join(data_dir, "AuxillaryTable.csv")
        self.theta = os.path.join(data_dir, "FM_Parameter_Table.csv")

        # List of transformations that will be applied in __getitem__()
        self.data_transforms: list[DataTransform] = []

        # Scaling transform for the parameters `theta` (e.g., minmax scaling)
        self.theta_scaler = (
            theta_scaler if theta_scaler is not None else IdentityScaler()
        )

        # Scaling transform for the auxiliary data `aux_data`
        self.auxiliary_data_scaler = (
            auxiliary_data_scaler if auxiliary_data_scaler is not None else AuxIdentityScaler()
        )


    def __getitem__(self, ind):

        # Load the data from the HDF5 file
        model = pd.read_hdf(self.fnames[ind])

        # Prepare the sample dict
        # TODO: only include requested components
        sample = {
            "theta" : self.get_parameters(indices=np.array(ind)).copy(),
            "wlen" : model['instrument_wlgrid'][:].to_numpy().copy(),
            "flux" : model['instrument_spectrum'][:].to_numpy().copy(),
            "error_bars" : model['instrument_noise'][:].to_numpy().copy(),
            "aux_data" : self.get_aux_data(indices=np.array(ind)).copy(),
        }

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
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")

    def get_parameters_labels(self) -> List[str]:
        assert self.theta.endswith('.csv'), "File must be a .csv file"
        
        try:
            theta = pd.read_csv(self.theta)
            labels = theta.columns.tolist()
            labels.remove('planet_ID')
            return labels
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        
    def get_parameters_dim(self) -> int:
        assert self.theta.endswith('.csv'), "File must be a .csv file"
        
        try:
            theta = pd.read_csv(self.theta)
            theta = theta.drop(columns=['planet_ID'])
            return theta.shape[1]
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        
    def get_aux_data_labels(self) -> List[str]:
        assert self.aux_data.endswith('.csv'), "File must be a .csv file"
        
        try:
            aux_data = pd.read_csv(self.aux_data)
            labels = aux_data.columns.tolist()
            labels.remove('planet_ID')
            return labels
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
        
    def get_aux_data_dim(self) -> int:
        assert self.aux_data.endswith('.csv'), "File must be a .csv file"
        
        try:
            aux_data = pd.read_csv(self.aux_data)
            aux_data = aux_data.drop(columns=['planet_ID'])
            return aux_data.shape[1]
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")
    
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

            # Convert units
            aux_data['star_radius_m'] = aux_data['star_radius_m'] / RSOL
            aux_data['planet_radius_m'] = aux_data['planet_radius_m'] / RJUP
            aux_data['planet_mass_kg'] = aux_data['planet_mass_kg'] / MJUP
            aux_data['star_mass_kg'] = aux_data['star_mass_kg'] / MSOL

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
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.aux_data} not found.")


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
        limit: int = None, 
        verbose: bool = True, 
        theta_scaler: ThetaScaler | None = None,
        auxiliary_data_scaler: AuxiliaryDataScaler | None = None
    ) -> tuple[INAFDataset, INAFDataset, INAFDataset]:
    """
    Load an INAF dataset from the given data directory.
    """

    # Load the train dataset
    train_dataset = INAFDataset(
        data_dir=data_dir,
        split='train',
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
    )
    
    #  Load the validation dataset
    val_dataset = INAFDataset(
        data_dir=data_dir,
        split='val',
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
    )
    # Load the test dataset
    test_dataset = INAFDataset(
        data_dir=data_dir,
        split='test',
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
    )


    return train_dataset, val_dataset, test_dataset
    

