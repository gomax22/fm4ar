# Write the code for the INARASubset class here
from functools import lru_cache

import os
import pickle
import numpy as np
import torch
import time
from typing import List
from pydantic import Field
from tqdm import tqdm

from torch.utils.data import Dataset
from fm4ar.datasets.data_transforms import DataTransform
from fm4ar.datasets.scalers.theta_scalers import ThetaScaler, IdentityScaler as ThetaIdentityScaler
from fm4ar.datasets.scalers.auxiliary_data_scalers import AuxiliaryDataScaler, IdentityScaler as AuxIdentityScaler
from fm4ar.datasets.scalers.flux_scalers import FluxScaler, IdentityScaler as FluxIdentityScaler
from fm4ar.datasets.scalers.error_bars_scalers import ErrorBarsScaler, IdentityScaler as ErrorBarsIdentityScaler
from fm4ar.datasets import DatasetConfig


NUM_INARA_DATASET_SAMPLES = 3_112_620 
NUM_INARA_SUBSET_SAMPLES = 91_392
DIM_THETA = 13
DIM_AUX = 15
DIM_WAVELENGTHS = 15_346

SECTOR_SIZE = 10_000
COMPONENTS = [
    'noise',
    'planet_signal',
    'wavelengths',
    'parameters',
    'star_planet_signal',
    'stellar_signal',
    'noise_profile'
]
THETA = {
    14: 'planet_surface_temperature_(Kelvin)',
    15: 'H2O',
    16: 'CO2',
    17: 'O2',
    18: 'N2',
    19: 'CH4',
    20: 'N2O',
    21: 'CO',
    22: 'O3',
    23: 'SO2',
    24: 'NH3',
    25: 'C2H6',
    26: 'NO2'
}

AUX_DATA = {
    1: 'star_class_(F=3_G=4_K=5_M=6)',
    2: 'star_temperature_(Kelvin)',
    3: 'star_radius_(Solar_radii)',
    4: 'distance_from_Earth_to_the_system_(parsec)',
    5: 'semimajor_axis_of_the_planet_(AU)',
    6: 'planet_radius_(Earth_radii)',
    7: 'planet_density_(g/cm3)',
    8: 'planet_surface_pressure_(bar)',
    9: 'kappa',
    10: 'gamma1',
    11: 'gamma2',
    12: 'alpha',
    13: 'beta',
    27: "planet_atmosphere's_avg_mol_wgt_(g/mol)",
    28: "planet's_mean_surface_albedo_(unitless)"
}
                

def get_sector(index: str) -> str:
    start = (int(index) // SECTOR_SIZE) * SECTOR_SIZE
    end   = start + SECTOR_SIZE
    return f"{start:07d}-{end:07d}"


def load_normalization_params(
    file_path: str,
) -> dict[str, np.ndarray]:
    """
    Load normalization parameters from an INARA dataset file.

    Args:
        file_path: Path to the INAF dataset file.

    Returns:
        A 2-tuple `(mean_flux, std_flux)`.
    """

    # Load the normalization parameters
    with open(file_path, "rb") as f:
        norm_params = pickle.load(f)

    # Extract the relevant entries e.g. only selected components, selected theta, selected aux.
    

    return norm_params 

class INARASubsetConfig(DatasetConfig):
    """
    Configuration for the INARA Subset.
    """
    data_dir: str = Field(
        ...,
        description="Path to the INARA Subset directory.",
    )
    targets: List[str] = Field(
        default=list(THETA.values()),
        description="List of target parameters to predict.",
    )
    components: List[str] = Field(
        default=COMPONENTS,
        description="List of dataset components to load.",
    )
    auxiliary_data: List[str] = Field(
        default=list(AUX_DATA.values()),
        description="List of auxiliary data parameters to load.",
    )
    limit: int | None = Field(
        ...,
        description="Limit the number of samples to load from the subset.",
    )
    verbose: bool = Field(
        ...,
        description="Whether to print verbose output during dataset loading.",
    )


class INARASubset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 split: str, 
                 targets: List[str] = list(THETA.values()),
                 components: List[str] = COMPONENTS,
                 auxiliary_data: List[str] = list(AUX_DATA.values()),
                 limit: int = None, 
                 verbose: bool = True,
                 theta_scaler: ThetaScaler | None = None,
                 auxiliary_data_scaler: AuxiliaryDataScaler | None = None,
                 flux_scaler: FluxScaler | None = None,
                 error_bars_scaler: ErrorBarsScaler | None = None
    ) -> None:
        super().__init__()
        assert limit is None or isinstance(limit, int), "Limit must be an integer or None."

        # TODO: add none option to split, adjust methods accordingly, remember that self.split_file could be None
        assert split in ["train", "val", "test", None], "Split must be one of 'train', 'val', 'test', or None."


        self.data_dir = data_dir
        self.split = split
        self.limit = limit
        self.verbose = verbose
        self.targets = {
            k:v for k, v in THETA.items() if v in targets
        } if targets is not None else THETA
        
        self.components = {
            comp for comp in components if comp in COMPONENTS
        } if components is not None else COMPONENTS
        
        self.auxiliary_data = {
            k:v for k, v in AUX_DATA.items() if v in auxiliary_data
        } if auxiliary_data is not None else AUX_DATA

        self.wavelengths = os.path.join(
            self.data_dir,
            "wavelengths.csv"
        )

        # load split file (used only for the mapping between index and planet index)
        self.split_file = self._load_data_split(split, limit)
        
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

    
    def __getitem__(self, ind) -> dict[str, torch.Tensor]:
        # integers iterate over entries of the dict / np.ndarray
        # string keys access specific entries (planet_index) / fpaths  

        # Prepare the sample dict
        # TODO: only include requested components
        # TODO: handle index as np.ndarray, List[str]
        sample = self.get_sample(ind)[0]

        # Apply the feature scaling for the flux
        sample = self.flux_scaler.forward(sample)

        # Apply the feature scaling for the error_bars
        sample = self.error_bars_scaler.forward(sample)

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
        return len(self.split_file) if self.split_file is not None else NUM_INARA_SUBSET_SAMPLES


    @property
    @lru_cache(maxsize=None)
    def dim_theta(self) -> int:
        """
        Return the number of parameters in the dataset, i.e., the
        dimensionality of `theta`.
        """

        # The dimensionality of the parameters should not be modified by any
        # of the transformations, so we do not need to apply them here.

        return len(self.targets)

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
        planet_index = self.split_file[0] if self.split_file is not None else 0
        return np.loadtxt(
            os.path.join(
                self.data_dir,
                self.split,
                "planet_signal",
                get_sector(planet_index),
                f"{planet_index}_planet_signal.csv"
            ),
            delimiter=',',
            dtype=np.float32
        ).shape[-1]
     

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

        return len(self.auxiliary_data)
    
    
    def get_wavelengths(self) -> np.ndarray:
        assert self.wavelengths.endswith('.csv'), "File must be a .csv file"
        
        try:

            # We load the wavelengths from the CSV file using np.loadtxt
            # for convenience, skipping the header row.
            wavelengths = np.loadtxt(self.wavelengths, skiprows=1)  # Skip header
            return wavelengths.astype(np.float32)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.wavelengths} not found.")

    
    def _load_data_split(self, split: str, limit: int = None) -> dict[int, str] | None:
        # Load split indices if split is specified
        try:
            split_file = np.load(
                os.path.join(
                    self.data_dir, 
                    f"{split}_indices.npy"
                )) if split is not None else None
        except FileNotFoundError:
            raise FileNotFoundError(f"Split file for {split} not found in {self.data_dir}.")
            
        # Apply limit to split file if specified
        if limit is not None and split_file is not None:
            assert isinstance(limit, int) and limit > 0, "Limit must be a positive integer for split file."
            if limit > len(split_file):
                if self.verbose: print(f"Limit {limit} exceeds available data size {len(split_file)} for split file. Using full split.")
            else:
                if self.verbose: print(f"Using first {limit} samples from {split} split for split file.")
                split_file = split_file[:limit]
        
        return {i: str(k).zfill(7) for i, k in enumerate(split_file)} \
            if split_file is not None \
            else {i: str(i).zfill(7) for i in range(NUM_INARA_SUBSET_SAMPLES)}


    def get_sample(
        self,
        index: int
    ) -> tuple[dict[str, np.ndarray], int]:
        """
        Get a single sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionary containing the sample data.
        """
        planet_index = self.split_file[index] if self.split_file is not None else index
        sector = get_sector(planet_index)

        sample = {
            "theta" : self.get_parameter_array(sector, planet_index).copy(),
            "wlen" : self.get_wavelengths().copy(),
            "aux_data" : self.get_aux_data_array(sector, planet_index).copy(),
            "flux": self.get_component_array('planet_signal', sector, planet_index).copy(),
            "error_bars":  self.get_component_array('noise', sector, planet_index).copy(),
        }

        return sample, planet_index
    

    def get_component(
        self,
        component: str,
    ) -> np.ndarray:
        """
        Get the file path for a specific component of the dataset.
        """

        data = []
        for pl_index in self.split_file.values():
            sector = get_sector(pl_index)
            comp = np.loadtxt(
                os.path.join(
                    self.data_dir,
                    self.split,
                    component,
                    sector,
                    f"{pl_index}_{component}.csv"
                ),
                delimiter=',',
                dtype=np.float32
            ) 
            data.append(comp)
        return np.array(data, dtype=np.float32)

    
    def get_component_array(self, component: str, sector: int, planet_index: str) -> np.ndarray:
        """
        Get the file path for a specific component of the dataset.
        """

        data = np.loadtxt(
            os.path.join(
                self.data_dir,
                self.split,
                component,
                sector,
                f"{planet_index}_{component}.csv"
            ),
            delimiter=',',
            dtype=np.float32
        ) 
        return np.array(data, dtype=np.float32)
    
    def get_parameters(
        self,
    ) -> np.ndarray:
        """
        Get the parameters `theta` of the dataset.
        """
        data = []
        for pl_index in self.split_file.values():
            sector = get_sector(pl_index)
            param = np.loadtxt(
                os.path.join(
                    self.data_dir,
                    self.split,
                    "theta",
                    sector,
                    f"{pl_index}_theta.csv"
                ),
                delimiter=',',
                skiprows=1,
                dtype=np.float32
            ) 
            data.append(param)
        return np.array(data, dtype=np.float32)

    def get_parameter_array(
        self, 
        sector: int, 
        planet_index: str, 
    ) -> np.ndarray:
        
        component = "theta"
        data = np.loadtxt(
            os.path.join(
                self.data_dir,
                self.split,
                component,
                sector,
                f"{planet_index}_{component}.csv"
            ),
            delimiter=',',
            skiprows=1,
            dtype=np.float32
        ) 

        # theta = []
        # for k in self.targets.keys():
        #     theta.append(data[k])
        # return np.array(theta, dtype=np.float32)

        return np.array(data, dtype=np.float32)

    def get_aux_data(
        self,
    ) -> np.ndarray:
        """
        Get the auxiliary data `aux_data` of the dataset.
        """
        data = []
        for pl_index in self.split_file.values():
            sector = get_sector(pl_index)
            aux = np.loadtxt(
                os.path.join(
                    self.data_dir,
                    self.split,
                    "aux_data",
                    sector,
                    f"{pl_index}_aux_data.csv"
                ),
                delimiter=',',
                skiprows=1,
                dtype=np.float32
            ) 
            data.append(aux)
        return np.array(data, dtype=np.float32)
    
    def get_aux_data_array(
        self,
        sector: int,
        planet_index: str,
    ) -> np.ndarray:
        
        component = "aux_data"
        data = np.loadtxt(
            os.path.join(
                self.data_dir,
                self.split,
                component,
                sector,
                f"{planet_index}_{component}.csv"
            ),
            delimiter=',',
            skiprows=1,
            dtype=np.float32
        ) 

        # extract only the requested auxiliary data columns
        # aux_data = []
        # for k in self.auxiliary_data.keys():
        # return np.array(aux_data, dtype=np.float32)

        return np.array(data, dtype=np.float32)
    
    def get_parameters_labels(self): 
        return list(THETA.values())
    
    def get_aux_data_labels(self): 
        return list(AUX_DATA.values())
    
    

def load_inara_dataset(
        data_dir: str, 
        targets: List[str] = list(THETA.values()),
        components: List[str] = COMPONENTS,
        auxiliary_data: List[str] = list(AUX_DATA.values()),
        limit: int = None, 
        verbose: bool = True, 
        theta_scaler: ThetaScaler | None = None,
        auxiliary_data_scaler: AuxiliaryDataScaler | None = None,
        flux_scaler: FluxScaler | None = None,
        error_bars_scaler: ErrorBarsScaler | None = None,
    ) -> tuple[INARASubset, INARASubset, INARASubset]:
    """
    Load an INAF dataset from the given data directory.
    """

    # Load the train dataset
    train_dataset = INARASubset(
        data_dir=data_dir,
        split='train',
        targets=targets,
        components=components,
        auxiliary_data=auxiliary_data,
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
        flux_scaler=flux_scaler,
        error_bars_scaler=error_bars_scaler,
    )
    
    #  Load the validation dataset
    val_dataset = INARASubset(
        data_dir=data_dir,
        split='val',
        targets=targets,
        components=components,
        auxiliary_data=auxiliary_data,
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
        flux_scaler=flux_scaler,
        error_bars_scaler=error_bars_scaler,
    )
    # Load the test dataset
    test_dataset = INARASubset(
        data_dir=data_dir,
        split='test',
        targets=targets,
        components=components,
        auxiliary_data=auxiliary_data,
        limit=limit,
        verbose=verbose,
        theta_scaler=theta_scaler,
        auxiliary_data_scaler=auxiliary_data_scaler,
        flux_scaler=flux_scaler,
        error_bars_scaler=error_bars_scaler,
    )

    return train_dataset, val_dataset, test_dataset
    



if __name__ == "__main__":
    from fm4ar.torchutils.dataloaders import build_dataloaders
    start_time = time.time()


    # Example usage
    data_dir = "/home/mgo/projects/data/PSG_INARA_Subset"
    tr_dataset = INARASubset(
        data_dir=data_dir,
        split='train',
        verbose=True
    )

    print("Train Dataset loaded in %.2f seconds." % (time.time() - start_time))
    print(f"Train Dataset size: {len(tr_dataset)}")
    sample = tr_dataset[0]

    start_time = time.time()
    vl_dataset = INARASubset(
        data_dir=data_dir,
        split='val',
        verbose=True
    )

    
    print("Valid Dataset loaded in %.2f seconds." % (time.time() - start_time))
    print(f"Valid Dataset size: {len(vl_dataset)}")
    sample = vl_dataset[0]

    train_loader, test_loader = build_dataloaders(
        train_dataset=tr_dataset,
        valid_dataset=vl_dataset,
        batch_size=256,
        n_workers=4,
        random_seed=42
    )
    print("DataLoaders built.")

    times = []

    for _ in tqdm(range(5), desc="Measuring batch loading time"):
        time_batch = time.time()
        batch = next(iter(train_loader))
        times.append(time.time() - time_batch)
    print(f"Average batch loading time: {np.mean(times):.3f} +- {np.std(times):.3f} seconds")
    del train_loader, test_loader