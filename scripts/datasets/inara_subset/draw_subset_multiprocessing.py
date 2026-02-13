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
from multiprocessing import Pool, cpu_count
from functools import partial
from dataclasses import dataclass

# --- CONSTANTS ---
NUM_INARA_SAMPLES = 3_112_620
SECTOR_SIZE = 10_000

COMPONENTS = [
    'noise', 'planet_signal', 'wavelengths', 'parameters',
    'star_planet_signal', 'stellar_signal',
]

THETA = {
    14: 'planet_surface_temperature_(Kelvin)', 15: 'H2O', 16: 'CO2', 17: 'O2',
    18: 'N2', 19: 'CH4', 20: 'N2O', 21: 'CO', 22: 'O3', 23: 'SO2',
    24: 'NH3', 25: 'C2H6', 26: 'NO2'
}

AUX_DATA = {
    1: 'star_class_(F=3_G=4_K=5_M=6)', 2: 'star_temperature_(Kelvin)',
    3: 'star_radius_(Solar_radii)', 4: 'distance_from_Earth_to_the_system_(parsec)',
    5: 'semimajor_axis_of_the_planet_(AU)', 6: 'planet_radius_(Earth_radii)',
    7: 'planet_density_(g/cm3)', 8: 'planet_surface_pressure_(bar)',
    9: 'kappa', 10: 'gamma1', 11: 'gamma2', 12: 'alpha', 13: 'beta',
    27: "planet_atmosphere's_avg_mol_wgt_(g/mol)",
    28: "planet's_mean_surface_albedo_(unitless)"
}

# --- STATS HELPER CLASS ---
@dataclass
class StatsState:
    """Stores running statistics for parallel aggregation."""
    count: int
    mean: np.ndarray
    M2: np.ndarray  # Sum of squares of differences from the current mean
    min_v: np.ndarray
    max_v: np.ndarray

    @staticmethod
    def create_empty(dims: int):
        return StatsState(
            count=0,
            mean=np.zeros(dims, dtype=np.float64),
            M2=np.zeros(dims, dtype=np.float64),
            min_v=np.full(dims, np.inf, dtype=np.float64),
            max_v=np.full(dims, -np.inf, dtype=np.float64)
        )

    def update(self, new_data: np.ndarray):
        """Update stats with a single new data point (Welford's algorithm)."""
        data = new_data.astype(np.float64)
        self.count += 1
        delta = data - self.mean
        self.mean += delta / self.count
        delta2 = data - self.mean
        self.M2 += delta * delta2
        self.min_v = np.minimum(self.min_v, data)
        self.max_v = np.maximum(self.max_v, data)

    def merge(self, other: 'StatsState'):
        """Merge another StatsState into this one (Parallel Welford)."""
        if other.count == 0:
            return

        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.M2 = other.M2
            self.min_v = other.min_v
            self.max_v = other.max_v
            return

        # Combine counts
        new_count = self.count + other.count

        # Combine means
        # Formula: Mean_X = (n_A * Mean_A + n_B * Mean_B) / (n_A + n_B)
        delta = other.mean - self.mean
        new_mean = self.mean + delta * (other.count / new_count)

        # Combine M2 (Sum of Squared Differences)
        # Formula: M2_X = M2_A + M2_B + delta^2 * (n_A * n_B) / (n_A + n_B)
        new_M2 = self.M2 + other.M2 + (delta ** 2) * (self.count * other.count / new_count)

        # Combine Min/Max
        new_min = np.minimum(self.min_v, other.min_v)
        new_max = np.maximum(self.max_v, other.max_v)

        self.count = new_count
        self.mean = new_mean
        self.M2 = new_M2
        self.min_v = new_min
        self.max_v = new_max

    def to_dict(self):
        """Convert to final dictionary format."""
        # Calculate Variance and Std
        if self.count > 1:
            variance = self.M2 / self.count # Population variance usually, or / (n-1) for sample
            # Keeping consistent with original script logic (numpy default std is population)
            std = np.sqrt(variance) 
        else:
            std = np.zeros_like(self.mean)
            
        return {
            "mean": self.mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.min_v.astype(np.float32),
            "max": self.max_v.astype(np.float32),
        }

# --- UTILS ---
def parse_components(arg):
    return arg.split(',')

def get_sector(index: int) -> str:
    start = (index // SECTOR_SIZE) * SECTOR_SIZE
    end = start + SECTOR_SIZE
    return f"{start:07d}-{end:07d}"

def get_parameters(parameters_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with open(parameters_dir) as f:
            entries = f.readlines()[2:]
    except FileNotFoundError:
        raise FileNotFoundError(f"File {parameters_dir} not found.")
    
    r_expr = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    data = []
    # Note: Sequential parsing here is fine as it happens once. 
    # Can be parallelized if needed, but usually acceptable startup cost.
    print("Parsing parameters file...")
    for line in tqdm(entries, desc="Parsing parameters"):
        numbers = re.findall(r_expr, line)
        data.append([float(num) for num in numbers])
        
    parameters = np.array(data, dtype=np.float32)
    theta = parameters[:, list(THETA.keys())[0] : list(THETA.keys())[0] + len(THETA)]
    aux_data = parameters[:, list(AUX_DATA.keys())[0] : list(AUX_DATA.keys())[0] + len(AUX_DATA)]
    planet_indices = parameters[:, 0].astype(np.int32)
    return theta, aux_data, planet_indices

def draw_indices(num_samples, train_fraction, val_fraction, test_fraction, seed, num_test_samples=None):
    # (Same logic as provided script)
    if num_test_samples is None:
        assert math.isclose(train_fraction + val_fraction + test_fraction, 1.0, rel_tol=1e-9)
    
    random.seed(seed)
    np.random.seed(seed)
    all_indices = np.arange(NUM_INARA_SAMPLES)
    sampled_indices = np.random.choice(all_indices, size=num_samples, replace=False)
    
    if num_test_samples is not None:
        test_indices = sampled_indices[:num_test_samples]
        remaining = sampled_indices[num_test_samples:]
        rem_count = num_samples - num_test_samples
        num_train = int(rem_count * train_fraction)
        train_indices = remaining[:num_train]
        val_indices = remaining[num_train:]
    else:
        num_train = int(num_samples * train_fraction)
        num_val = int(num_samples * val_fraction)
        train_indices = sampled_indices[:num_train]
        val_indices = sampled_indices[num_train:num_train + num_val]
        test_indices = sampled_indices[num_train + num_val:]
        
    return train_indices, val_indices, test_indices

# --- WORKER FUNCTION ---
def process_batch(args):
    """
    Worker function to process a batch of indices.
    Args tuple contains: (indices, component, split, data_dir, output_dir, preloaded_data)
    """
    indices, component, split, data_dir, output_dir, preloaded_data = args
    
    # Initialize local stats
    # Determine dimensions from first item (hacky but effective) or predefined
    dims = 0
    local_stats = None 
    
    for idx in indices:
        sector = get_sector(idx)
        dest_dir = output_dir / split / component / sector
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        data_values = None

        if component in ['theta', 'aux_data']:
            # Memory -> CSV
            # preloaded_data is passed as a dict {index: row_array} or a slice if carefully managed.
            # However, passing huge dicts to workers is slow. 
            # Ideally, preloaded_data is the full array and we access by index if using SharedMemory,
            # but for simplicity in multiprocessing, we assume 'preloaded_data' is a subset passed to this worker.
            
            # To avoid pickling huge arrays, 'preloaded_data' here receives ONLY the rows for this batch.
            # Structure: preloaded_data is a numpy array of shape (len(indices), dims)
            
            # Find the row index relative to the batch
            # We assume the caller passed the exact data chunk corresponding to `indices`
            batch_internal_idx = np.where(indices == idx)[0][0] 
            data_values = preloaded_data[batch_internal_idx]
            
            col_map = THETA if component == 'theta' else AUX_DATA
            df = pd.DataFrame(data_values.reshape(1, -1), columns=list(col_map.values()))
            df.to_csv(dest_dir / f"{idx:07d}_{component}.csv", index=False)

        else:
            # File -> File + Load
            src_file = data_dir / component / sector / f"{idx:07d}_{component}.csv"
            shutil.copy(src_file, dest_dir / src_file.name)
            
            # Load for stats
            try:
                data_values = np.loadtxt(src_file, delimiter=',', dtype=np.float32)
            except Exception as e:
                print(f"Error reading {src_file}: {e}")
                continue

        # Init stats on first valid data
        if local_stats is None and data_values is not None:
            dims = data_values.shape[0] if data_values.ndim == 1 else data_values.shape[1]
            local_stats = StatsState.create_empty(dims)

        # Update stats
        if data_values is not None:
            local_stats.update(data_values)

    return local_stats

# --- MAIN ---
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
    random.seed(seed)
    np.random.seed(seed)

    components = list(set(components + ['theta', 'aux_data']) - {'wavelengths', 'parameters'})
    print(f"Components: {components}")

    # Draw Indices
    train_idx, val_idx, test_idx = draw_indices(
        num_samples, train_fraction, val_fraction, test_fraction, seed, num_test_samples
    )
    
    print(f"Samples: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "val_indices.npy", val_idx)
    np.save(output_dir / "test_indices.npy", test_idx)

    # Load Global Parameters Once
    full_theta, full_aux, _ = get_parameters(data_dir / 'parameters' / 'parameters.tbl')
    
    # Global Stats Storage
    final_stats = {c: {s: None for s in ["train", "val", "test"]} for c in components}

    # Processing Loop
    splits = [("train", train_idx), ("val", val_idx), ("test", test_idx)]
    
    # Batch size for multiprocessing
    BATCH_SIZE = 1000 
    NUM_WORKERS = max(1, cpu_count() - 2) # Leave 1-2 cores for system/main thread

    for component in components:
        print(f"\n--- Processing {component} ---")
        
        for split_name, indices in splits:
            if len(indices) == 0: continue
            
            print(f"Processing {split_name} ({len(indices)} samples)...")
            
            # Prepare chunks
            chunks = []
            for i in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[i : i + BATCH_SIZE]
                
                # If component is memory-based, slice the data now to pass small bits to workers
                batch_data = None
                if component == 'theta':
                    batch_data = full_theta[batch_indices]
                elif component == 'aux_data':
                    batch_data = full_aux[batch_indices]
                
                chunks.append((batch_indices, component, split_name, data_dir, output_dir, batch_data))

            # Run Parallel
            # Use imap_unordered for progress bar
            split_stats_agg = None
            
            with Pool(processes=NUM_WORKERS) as pool:
                # Wrap process_batch with partial if needed, but here simple tuple arg is fine
                results = list(tqdm(
                    pool.imap_unordered(process_batch, chunks),
                    total=len(chunks),
                    desc=f"{component} [{split_name}]"
                ))
            
            # Aggregate Results
            print("Aggregating statistics...")
            for res in results:
                if res is None: continue
                
                if split_stats_agg is None:
                    split_stats_agg = res # First valid result becomes base
                else:
                    split_stats_agg.merge(res)
            
            # Save stats for this split
            if split_stats_agg:
                final_stats[component][split_name] = split_stats_agg.to_dict()

    # Save Stats
    with open(output_dir / "norm_params.pkl", "wb") as f:
        pickle.dump(final_stats, f)
    print(f"\nSaved normalization parameters.")

    # Copy wavelengths
    try:
        shutil.copy(data_dir / "wavelengths.csv", output_dir / "wavelengths.csv")
    except Exception as e:
        print(f"Warning: Could not copy wavelengths.csv: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--num-test-samples", type=int, default=None) # Changed default to None for logic consistency
    ap.add_argument("--components", type=parse_components, default=",".join(COMPONENTS))
    ap.add_argument("--seed", type=int, default=42)
    
    args = vars(ap.parse_args())
    draw_subset(**args)
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")