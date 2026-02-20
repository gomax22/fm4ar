import torch
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from time import time
from tqdm import tqdm

from fm4ar.utils.paths import expand_env_variables_in_path

def measure_time(
    experiment_dir: Path,
    log_file: str,
) -> tuple[str, int]:
    
    with (experiment_dir / log_file).open("r") as f:
        lines = f.readlines()

    # Grep all the lines that contain "This took"
    # For each line, extract the time in seconds
    # Get the total training time by summing all the times
    times = []

    for line in lines:
        if "This took" in line:
            # Extract the time in seconds
            parts = line.strip().split(" ")
            time_seconds = float(parts[-2].replace(",", ""))
            times.append(time_seconds)

    total_time = sum(times)
    
    # Exceptional cases
    if total_time == 0:
        times = []
        for line in lines:
            if "t_model" in line:
                # Extract the time in seconds
                parts = line.strip().split(" ")
                t_model_time = float(parts[-4])
                t_data_time = float(parts[-13])
                times.append(t_model_time + t_data_time)
        total_time = sum(times) / 1e3  # Convert ms to s

    return total_time

def measure_experimental_times(
    root_dir: Path,
) -> pd.DataFrame:
    rows = []

    root = Path(root_dir)

    # Traverse recursively
    experiment_dirs = tqdm(list(root.glob("*/*/*")), desc="Processing experiments")


    # TODO: Parallelize this loop if needed
    for experiment_dir in experiment_dirs:
        training_log = experiment_dir / "training.txt"
        if not training_log.exists():
            continue

        training_time = measure_time(
            experiment_dir=experiment_dir,
            log_file="training.txt",
        )

        sampling_time = measure_time(
            experiment_dir=experiment_dir,
            log_file="sampling.txt",
        )

        rows.append({
            "experiment_name": experiment_dir.name,
            "experiment_dir": str(experiment_dir),
            "training_time_seconds": training_time,
            "sampling_time_seconds": sampling_time,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    script_start = time()
    print("\nMEASURE EXPERIMENTAL TIMES\n")
    print("\nWARNING: This script assumes that the stdout / stderr during training and sampling " \
          "are redirected to `training.txt` and `sampling.txt` respectively, which is the case when using the " \
          "scripts in `scripts/train/` and `scripts/sampling/`. If you are using a different setup, " \
          "you may need to modify this script accordingly.\n"
    )

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the experiments directory.",
    )
    args = parser.parse_args()


    root_dir = expand_env_variables_in_path(args.root_dir)
    print("Root directory:", flush=True)
    print(f"{root_dir.resolve()}\n", flush=True)

    df = measure_experimental_times(
        root_dir=root_dir,
    )

    # Save to CSV
    output_csv = root_dir / "experiment_times.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved experiment info to {output_csv.resolve()}\n", flush=True)

    print(f"\nThis took {time() - script_start:.2f} seconds!\n", flush=True)
        

