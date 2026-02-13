
import argparse
import pickle
import random
import shutil
import time
import re
import numpy as np
import math

import pandas as pd
from tqdm import tqdm
from pathlib import Path


NUM_INARA_SAMPLES = 3_112_620 
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

def parse_components(arg):
    return arg.split(',')


def get_parameters(
    parameters_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: 
    
    # Load parameters
    try:
        with open(parameters_dir) as f:
            entries = f.readlines()[2:]
    except FileNotFoundError:
        raise FileNotFoundError(f"File {parameters_dir} not found.")
        
    # parse lines
    # columns = [line.strip() for line in lines[1].split('|') if line != ''][:-1]
    r_expr = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    
    data = []

    # WARNING: 3M entries, this is slow. 
    for line in entries:
        # Extract numbers from the line using regex
        numbers = re.findall(r_expr, line)
        data.append([float(num) for num in numbers])  

    parameters = np.array(data, dtype=np.float32)
    
    # Extract theta parameters
    theta = parameters[:, list(THETA.keys())[0] : list(THETA.keys())[0] + len(THETA)]

    # Extract auxiliary data parameters
    aux_data = parameters[:, list(AUX_DATA.keys())[0] : list(AUX_DATA.keys())[0] + len(AUX_DATA)]

    # Extract planet indices
    planet_indices = parameters[:, 0].astype(np.int32)

    return theta, aux_data, planet_indices


# Welford's algorithm for computing running stats 
def collect_stats(
    data_dir: Path,
    output_dir: Path,
    indices: list[int],
    component: str,
    split: str,
    dims: int,
) -> dict[str, np.ndarray]: 

    mean = np.zeros(dims)
    M2 = np.zeros(dims)
    minv = np.full(dims, np.inf)
    maxv = np.full(dims, -np.inf)

    mean_hist = np.zeros((len(indices), dims))
    std_hist  = np.zeros((len(indices), dims))
    min_hist  = np.zeros((len(indices), dims))
    max_hist  = np.zeros((len(indices), dims))

    pbar = tqdm(total=len(indices), desc=f"Copying {split} data and collecting stats for {component}")
    for t, index in enumerate(indices):
        
        n = t + 1
        sector = get_sector(index)
        if component in ['noise', 'planet_signal', 'stellar_signal', 'star_planet_signal']:
            src_file = data_dir / component / sector / f"{index:07d}_{component}.csv"
            dest_dir = output_dir / split / component / sector
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dest_dir / src_file.name)

            # Access the data to update stats
            data = np.loadtxt(src_file, delimiter=',', dtype=np.float32)
            
        elif component == 'theta':
            theta = [0]
            
            # extract theta for this index
            data = theta[index, :]
            dest_dir = output_dir / split / component / sector
            dest_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(data.reshape(1, -1), columns=list(THETA.values()))
            df.to_csv(dest_dir / f"{str(index).zfill(7)}_{component}.csv", index=False)

        elif component == 'aux_data':
            aux_data = get_parameters(data_dir / 'parameters' / 'parameters.tbl')[1]

            # extract aux_data for this index
            data = aux_data[index, :]
            dest_dir = output_dir / split / component / sector
            dest_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(data.reshape(1, -1), columns=list(AUX_DATA.values()))
            df.to_csv(dest_dir / f"{str(index).zfill(7)}_{component}.csv", index=False)

        else:
            raise ValueError(f"Unknown component: {component}")

                
        # --- mean & std (Welford) ---
        data = data.astype(np.float32)
        delta = data - mean
        mean += delta / n
        delta2 = data - mean
        M2 += delta * delta2

        if n > 1:
            std = np.sqrt(M2 / n)
        else:
            std = np.zeros(dims)

        # --- min & max ---
        minv = np.minimum(minv, data)
        maxv = np.maximum(maxv, data)

        mean_hist[t] = mean
        std_hist[t]  = std
        min_hist[t]  = minv
        max_hist[t]  = maxv

        pbar.update(1)
    pbar.close()

    return {
        "mean": mean_hist,
        "std":  std_hist,
        "min":  min_hist,
        "max":  max_hist
    }
                


def get_sector(index: int) -> int:
    start = (index // SECTOR_SIZE) * SECTOR_SIZE
    end   = start + SECTOR_SIZE
    return f"{start:07d}-{end:07d}"


def draw_indices(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    num_test_samples: int = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Draw random sample indices without replacement for each split.
    """
    if num_test_samples is None:
        assert math.isclose(train_fraction + val_fraction + test_fraction, 1.0, rel_tol=1e-9), \
            "Fractions must sum to 1.0"
    assert num_samples <= NUM_INARA_SAMPLES, \
        f"Number of samples ({num_samples}) in the subset must be less than or equal to \
        the total number of samples in the dataset ({NUM_INARA_SAMPLES})"
    assert num_test_samples is None or num_test_samples <= num_samples, \
        "Number of test samples must be less than or equal to the total number of samples"

    random.seed(seed)
    np.random.seed(seed)

    all_indices = np.arange(NUM_INARA_SAMPLES)
    sampled_indices = np.random.choice(
        all_indices, size=num_samples, replace=False
    )
    if num_test_samples is not None:
        test_indices = sampled_indices[:num_test_samples]
        remaining_indices = sampled_indices[num_test_samples:]
        remaining_num_samples = num_samples - num_test_samples

        num_train = int(remaining_num_samples * train_fraction)
        num_val = remaining_num_samples - num_train
        train_indices = remaining_indices[:num_train]
        val_indices = remaining_indices[num_train:]

    else:

        num_train = int(num_samples * train_fraction)
        num_val = int(num_samples * val_fraction)
        num_test = num_samples - num_train - num_val
        train_indices = sampled_indices[:num_train]
        val_indices = sampled_indices[num_train:num_train + num_val]
        test_indices = sampled_indices[num_train + num_val:]

    return train_indices, val_indices, test_indices

def draw_subset(
    data_dir: Path,
    output_dir: Path,
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    num_test_samples: int = None,
    components: list[str] = COMPONENTS,
):
    """
    Draw a subset of the INARA dataset and save it to the output directory.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Define components to copy
    components = list(
        set(components + ['theta', 'aux_data']) - {'wavelengths', 'parameters'}
    )
    print(f"Components to be processed: {components} ({len(components)})")

    if num_test_samples is not None:
        print(f"Number of test samples fixed to: {num_test_samples}")


    # Draw indices
    train_indices, val_indices, test_indices = draw_indices(
        num_samples,
        train_fraction,
        val_fraction,
        test_fraction,
        seed,
        num_test_samples
    )
    print(f"\nNumber of samples drawn: {num_samples:_} out of {NUM_INARA_SAMPLES:_} ({num_samples / NUM_INARA_SAMPLES:.2%})")

    test_fraction = num_test_samples / num_samples
    train_fraction = 0.8 * (1 - test_fraction)
    val_fraction = 0.2 * (1 - test_fraction)

    print(f"Train fraction: {train_fraction:.2%} ({len(train_indices):_} samples)")
    print(f"Validation fraction: {val_fraction:.2%} ({len(val_indices):_} samples)")
    print(f"Test fraction: {test_fraction:.2%} ({len(test_indices):_} samples)\n")


    # save the indices
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(output_dir / "val_indices.npy", val_indices)
    np.save(output_dir / "test_indices.npy", test_indices)
    print(f"Saved train/val/test indices to {output_dir}\n")


    # Initialize stats dictionary
    stats = {
        component: {
            split: None for split in ["train", "val", "test"]
        } for component in components
    }

    # For each component and each split, copy the relevant files
    # Retrieve the data and copy files

    for component in components:
        print(f"--- Processing component: {component} ---")

        # TODO: add multithreading / multiprocessing for copying files and collecting stats, 
        # especially for noise and signals which are the largest components.
        for split, indices in zip(
            ["train", "val", "test"],
            [train_indices, val_indices, test_indices]
        ):
            print("\n###############################################")
            print(f"# Processing {split} split with {len(indices):_} samples. #")
            print("###############################################\n")
        
            # Load splits once per component
            if component == 'theta':
                component_data, _, _ = get_parameters(data_dir / 'parameters' / 'parameters.tbl')
                print(f"Loaded theta parameters.")

                # Save theta files
                # extract theta for this index
                pbar = tqdm(total=len(indices), desc=f"Copying theta data for {split} split")
                for index in indices:
                    sector = get_sector(index)
                    data = component_data[index, :]
                    dest_dir = output_dir / split / component / sector
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    df = pd.DataFrame(data.reshape(1, -1), columns=list(THETA.values()))
                    df.to_csv(dest_dir / f"{index:07d}_{component}.csv", index=False)
                    pbar.update(1)
                pbar.close()

            elif component == 'aux_data':
                _, component_data, _ = get_parameters(data_dir / 'parameters' / 'parameters.tbl')
                print(f"Loaded auxiliary data parameters.")

                # Save aux_data files
                pbar = tqdm(total=len(indices), desc=f"Copying aux_data for {split} split")
                for index in indices:
                    sector = get_sector(index)
                    data = component_data[index, :]
                    dest_dir = output_dir / split / component / sector
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    df = pd.DataFrame(data.reshape(1, -1), columns=list(AUX_DATA.values()))
                    df.to_csv(dest_dir / f"{index:07d}_{component}.csv", index=False)
                    pbar.update(1)
                pbar.close()
            else:
                
                component_data = []

                pbar = tqdm(total=len(indices), desc=f"Copying data for {component} in {split} split")
                for index in indices:
                    sector = get_sector(index)
                    src_file = data_dir / component / sector / f"{index:07d}_{component}.csv"
                    dest_dir = output_dir / split / component / sector
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_file, dest_dir / src_file.name)

                    # Access the data to update stats
                    component_data.append(
                        np.loadtxt(src_file, delimiter=',', dtype=np.float32))
                    pbar.update(1)
                pbar.close()

                component_data = np.array(component_data, dtype=np.float32)

                
            # Collect stats
            stats[component][split] = {
                "mean": np.mean(component_data, axis=0),
                "std":  np.std(component_data, axis=0),
                "min":  np.min(component_data, axis=0),
                "max":  np.max(component_data, axis=0),
            }


    # Save stats to output directory
    with open(output_dir / "norm_params.pkl", "wb") as f:
        pickle.dump(stats, f)
        print(f"Saved normalization parameters to {output_dir / 'norm_params.pkl'}")

    # Make a copy of the wavelengths file
    wavelengths_src = data_dir / "wavelengths.csv"
    wavelengths_dest = output_dir / "wavelengths.csv"
    shutil.copy(wavelengths_src, wavelengths_dest)
    print(f"Copied wavelengths file to {wavelengths_dest}")

    print("Done.")


# WARNING: This script is memory-intensive because it duplicates a subset of the dataset (potentially a perfect copy).
# Before running the script, ensure you have enough space available.
if __name__ == "__main__":
    start_time = time.time()

    ap = argparse.ArgumentParser(description="Draw a subset of INARA dataset.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the INARA dataset directory",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the output directory where drawn subset will be saved",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of samples to draw from the dataset",
    )
    ap.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of samples to allocate to the training set",
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples to allocate to the validation set",
    )    
    ap.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of samples to allocate to the test set",
    )
    ap.add_argument(
        "--num-test-samples",
        type=int,
        default=9140,
        help="Number of samples to draw for the test set",
    )
    ap.add_argument(
        "--components",
        type=parse_components,
        default=",".join(COMPONENTS),
        help=f"Comma-separated list of components to include in the subset (default: all components). \
              Parameters (target, auxiliary data) and wavelengths are always included. \
              Available components: {', '.join(COMPONENTS)}",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = vars(ap.parse_args())

    draw_subset(**args)

    print(f"This took {time.time() - start_time:.2f} seconds")