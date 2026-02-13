import os
import gc
import time
import pickle
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from typing import List
from tqdm import tqdm
from pprint import pprint

from fm4ar.datasets.inaf import INAFDataset

COMPONENTS = [
    'instrument_spectrum',
    'instrument_noise',
    'instrument_wlgrid',
    'instrument_width',
    'theta',
    'aux_data'
]

def draw_indices(
    num_samples: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> List[np.ndarray]:
    """
    Draw random indices for train, validation and test splits.
    
    Parameters
    ----------
    num_samples : int
        Total number of samples in the dataset.
    train_fraction : float, optional
        Fraction of samples to use for training, by default 0.7
    val_fraction : float, optional
        Fraction of samples to use for validation, by default 0.2
    test_fraction : float, optional
        Fraction of samples to use for testing, by default 0.1
    
    Returns
    -------
    List[np.ndarray]
        A list containing the train, validation and test indices.
    """
    
    assert np.allclose(np.sum([train_fraction, val_fraction, test_fraction]), 1.0), "Train, val and test fractions must sum to 1.0"
    assert train_fraction > 0.0, "Train fraction must be greater than 0.0"
    assert val_fraction > 0.0, "Validation fraction must be greater than 0.0"
    assert test_fraction > 0.0, "Test fraction must be greater than 0.0"
    
    print(f"Number of samples found: {num_samples}")

    # get train, val, test indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Compute split sizes
    train_end = int(train_fraction * num_samples)
    val_end = train_end + int(val_fraction * num_samples)

    # Slice indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices


def generate_real_dataset(
    dataset: INAFDataset,
    extended_comps: List[str],
    indices: np.ndarray,
    norm_params: dict,
    output_dir: str,
    mode: str = 'train',
) -> None:
    """
    Generate the real dataset by adding noise to the instrument spectrum and store it in HDF5 format.
    
    Parameters
    ----------
    dataset : INAFDataset
        The INAF dataset object.
    extended_comps : List[str]
        List of components to include in the dataset, including the additional components for real spectrum and noise profile.
    indices : np.ndarray
        The indices of the samples to include in the dataset split.
    norm_params : dict
        Dictionary to store the normalization parameters for each component and split.
    output_dir : str
        The directory to save the generated dataset.
    mode : str, optional
        The dataset split mode (train, val, test), by default 'train'
    
    Returns
    -------
    dict
        Updated normalization parameters for the components in the dataset split.
    """


    # Create the output directory if it doesn't exist
    h5s_output_dir = os.path.join(output_dir, "h5s")
    if not os.path.exists(h5s_output_dir):
        os.makedirs(output_dir, exist_ok=True)

    components, additional_comps = extended_comps[:-2], extended_comps[-2:]

    # get normalization parameters for aux_data and theta
    for comp, fn in zip(['aux_data', 'theta'], [dataset.get_aux_data, dataset.get_parameters]):
        data = fn(indices=indices, skip_planet_index_col=True)
        norm_params[comp][mode]['min'] = data.min(axis=0).astype(np.float32)
        norm_params[comp][mode]['max'] = data.max(axis=0).astype(np.float32)
        norm_params[comp][mode]['mean'] = data.mean(axis=0).astype(np.float32)
        norm_params[comp][mode]['std'] = data.std(axis=0).astype(np.float32)

    # Create a progress bar
    pbar = tqdm(total=len(indices), desc=f"Generating real spectra for {mode} split", unit="spectrum")
    for i, idx in enumerate(indices):
        fname = dataset.fnames[idx]
        model = pd.read_hdf(fname)
        sample = dataset[idx]['sample']

        fname = fname.split(os.sep)[-1]

        for sample_key in sample.keys():
            sample[sample_key] = sample[sample_key].numpy().astype(np.float64)

        # Extract the spectrum and noise
        sample['instrument_spectrum'] = sample['flux']
        sample['instrument_noise'] = sample['error_bars']
        sample['instrument_wlgrid'] = sample['wlen']
        sample['instrument_width'] = model['instrument_width'][:].to_numpy().copy()
        del sample['flux'], sample['error_bars'], sample['wlen']

        # Generate the real spectrum
        n_profile = pd.Series(np.random.normal(loc=0, scale=sample['instrument_noise'], size=sample['instrument_noise'].shape))
        real_spectrum = pd.Series(sample['instrument_spectrum'] + n_profile)

        # Store the real spectrum
        df = model.assign(real_spectrum=real_spectrum, noise_profile=n_profile)        
        df.to_hdf(os.path.join(h5s_output_dir, fname), key=f'Planet_{idx}', mode='w', format='table', data_columns=True, index=False)
        
        # Update the normalization parameters
        for comp in components:  
            if comp in ['aux_data', 'theta']:
                continue
            
            out_sample = sample[comp].reshape(1, -1).astype(np.float32)
            if i == 0:
                norm_params[comp][mode]["min"] = out_sample
                norm_params[comp][mode]["max"] = out_sample
            else:
                
                norm_params[comp][mode]["min"] = np.min(np.concatenate([norm_params[comp][mode]["min"].reshape(1,-1), out_sample], axis=0), axis=0)
                norm_params[comp][mode]["max"] = np.max(np.concatenate([norm_params[comp][mode]["max"].reshape(1,-1), out_sample], axis=0), axis=0)
            
        for comp, data in zip(additional_comps, [real_spectrum, n_profile]):
            out_data = data.to_numpy().reshape(1, -1).astype(np.float32)
            if i == 0:
                norm_params[comp][mode]["min"] = out_data
                norm_params[comp][mode]["max"] = out_data
            else:
                norm_params[comp][mode]["min"] = np.min(np.concatenate([norm_params[comp][mode]["min"].reshape(1,-1), out_data], axis=0), axis=0)
                norm_params[comp][mode]["max"] = np.max(np.concatenate([norm_params[comp][mode]["min"].reshape(1,-1), out_data], axis=0), axis=0)
        pbar.update(1)


    pbar.close()

    # Close the HDF5 file
    gc.collect()
    return norm_params



def make_dataset(
    data_dir: str,
    output_dir: str,
    predict_log_temperature: bool,
    components: List[str],
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> None:
    """
    Make the INAF dataset suitable for training by drawing random indices for train, validation and test splits, generating the real dataset by adding noise to the instrument spectrum and storing it in HDF5 format, and computing the normalization parameters for each component and split.
    
    Parameters
    ----------
    data_dir : str
        Path to the training data.
    output_dir : str
        Path to save the generated dataset and normalization parameters.
    predict_log_temperature : bool
        Whether to predict log temperature instead of temperature.
    components : List[str]
        List of components to include in the dataset.
    train_fraction : float, optional
        Fraction of data to use for training, by default 0.7
    val_fraction : float, optional
        Fraction of data to use for validation, by default 0.2
    test_fraction : float, optional
        Fraction of data to use for testing, by default 0.1
    
    Returns
    -------
    None
    """

    
    print(components)
    dataset = INAFDataset(
        data_dir=data_dir,
        predict_log_temperature=predict_log_temperature,
        split=None,
        limit=None,
        verbose=True
    )
    train_indices, val_indices, test_indices = draw_indices(
        num_samples=len(dataset),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction
    )
    # Store indices
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(output_dir, "val_indices.npy"), val_indices)
    np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)

    additional_comps = ['real_spectrum', 'noise_profile']
    extended_comps = components + additional_comps

    norm_params = {
        comp: {
            split: {
                "mean": 0.0,
                "std": 0.0,
                "min": float("inf"),
                "max": float("-inf")
            } for split in ["train", "val", "test"]
        } for comp in extended_comps
    }
    # Generate the dataset and normalization parameters for min max normalization
    norm_params = generate_real_dataset(dataset, extended_comps, train_indices, norm_params, output_dir, mode='train')
    norm_params = generate_real_dataset(dataset, extended_comps, val_indices, norm_params, output_dir, mode='val')
    norm_params = generate_real_dataset(dataset, extended_comps, test_indices, norm_params, output_dir, mode='test')

    # get the normalization parameters for aux_data and theta
    norm_params_file_name = f"norm_params.pkl" if not predict_log_temperature else f"norm_params_with_log_temperature.pkl"
    norm_params_pkl_path = os.path.join(output_dir, norm_params_file_name)
    with open(norm_params_pkl_path, "wb") as f:
        pickle.dump(norm_params, f)
    print(f"Normalization parameters saved to {norm_params_pkl_path}")
    pprint(norm_params)

# --- UTILS ---
def parse_components(arg):
    return arg.split(',')

if __name__ == '__main__':
    start_time = time.time()

    ap = ArgumentParser(description='Make the SBI_INAF dataset suitable for training')
    ap.add_argument('--data_dir', type=str, required=True, help='Path to the training data')
    ap.add_argument('--predict_log_temperature', action='store_true', help='Whether to predict log temperature instead of temperature')
    ap.add_argument('--components', type=parse_components, default=",".join(COMPONENTS), help='List of components to include in the dataset')
    ap.add_argument('--train_fraction', type=float, default=0.7, help='Fraction of data to use for training')
    ap.add_argument('--val_fraction', type=float, default=0.2, help='Fraction of data to use for validation')
    ap.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of data to use for testing')
    ap.add_argument('--output_dir', type=str, required=True, help='Path to save the dataset')
    args = ap.parse_args()
    args = vars(args)
    make_dataset(**args)

     
    print(f"Total time: {time.time() - start_time:.2f} seconds")
