"""
Instantiate a model from a config file and count the number of
parameters without starting a training run.
"""
import torch
import pandas as pd

from typing import Optional
from argparse import ArgumentParser
from pathlib import Path
from time import time
from tqdm import tqdm

from fm4ar.datasets import load_dataset
from fm4ar.models.build_model import build_model
from fm4ar.models.fmpe import FMPENetwork
from fm4ar.models.npe import NPENetwork
from fm4ar.torchutils.general import get_number_of_parameters, get_flops, move_batch_to_device
from fm4ar.torchutils.dataloaders import build_dataloader, get_number_of_workers
from fm4ar.utils.paths import expand_env_variables_in_path
from fm4ar.utils.config import load_config



def count_parameters_and_flops(
    experiment_dir: Path,
    checkpoint_name: Optional[str] = None,
) -> tuple[str, int, int]:
    
    experiment_name = experiment_dir.name

    # Load model
    config = load_config(experiment_dir)


    # Load the dataset (using config from checkpoint)
    _, _, test_dataset = load_dataset(config)

    # Build dataloader for the test dataset
    test_dataloader = build_dataloader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        n_workers=get_number_of_workers(config['training']['stage_0']['n_workers']),
    )

    # Add the theta_dim and context_dim to the model settings
    config["model"]["dim_theta"] = test_dataset.dim_theta
    config["model"]["dim_context"] = test_dataset.dim_context
    config["model"]["dim_auxiliary_data"] = test_dataset.dim_auxiliary_data

    # Instantiate the posterior model
    file_path = experiment_dir / checkpoint_name if checkpoint_name else None
    model = build_model(
        experiment_dir=experiment_dir,
        file_path=file_path,
        device=config["local"]["device"],
    )

    # Count parameters
    n_total_params = get_number_of_parameters(model.network)

    batch = next(iter(test_dataloader))
    theta, context, aux_data = move_batch_to_device(batch, model.device)
    t = torch.zeros((theta.shape[0],), device=model.device) 

    # Compute FLOPs
    if isinstance(model.network, FMPENetwork):
        inputs = (t, theta, context, aux_data)
    elif isinstance(model.network, NPENetwork):
        inputs = (theta, context, aux_data)
    else:
        raise ValueError(f"Unknown network type: {type(model.network)}")
    
    flops = get_flops(model.network, inputs) 
    return experiment_name, n_total_params, flops

def collect_experiment_info(
    root_dir: Path,
    checkpoint_name: str = "model__best.pt",
) -> pd.DataFrame:
    """
    Scans experiment directories, loads each model__best.pt,
    and collects metadata into a dataframe.

    Args:
        root_dir (Path): Root directory containing experiment subdirectories.
        checkpoint_name (str): Name of the model checkpoint file.

    Returns:
        pd.DataFrame
    """
    rows = []

    root = Path(root_dir)

    # Traverse recursively
    experiment_dirs = tqdm(list(root.glob("*/*/*")), desc="Processing experiments")


    # TODO: Parallelize this loop if needed
    for experiment_dir in experiment_dirs:
        ckpt = experiment_dir / checkpoint_name
        if not ckpt.exists():
            continue

        experiment_name, n_total_params, flops = count_parameters_and_flops(
            experiment_dir=experiment_dir,
            checkpoint_name=checkpoint_name,
        )

        rows.append({
            "experiment_name": experiment_name,
            "experiment_dir": str(experiment_dir),
            "num_params_M": n_total_params / 1e6,
            "flops_G": flops / 1e9,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    script_start = time()
    print("\nCOUNT PARAMETERS AND FLOPS FOR EXPERIMENTS\n")

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the experiments directory.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Name of the model checkpoint file.",
    )
    args = parser.parse_args()


    root_dir = expand_env_variables_in_path(args.root_dir)
    print("Root directory:", flush=True)
    print(f"{root_dir.resolve()}\n", flush=True)

    df = collect_experiment_info(
        root_dir=root_dir,
        checkpoint_name=args.checkpoint_name,
    )

    # Save to CSV
    output_csv = root_dir / "experiment_parameters_and_flops.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved experiment info to {output_csv.resolve()}\n", flush=True)

    print(f"\nThis took {time() - script_start:.2f} seconds!\n", flush=True)
        

