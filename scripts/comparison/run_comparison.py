import argparse
from time import time
import pandas as pd

from typing import Optional
from pathlib import Path
from yaml import safe_load
from tqdm import tqdm

from fm4ar.utils.paths import expand_env_variables_in_path

# for each experiment:
    # load regression, calibration, coverage, log probs results
    # merge results into a single dataframe per each task
    # load predictions
    # make plots (corner, tarp, regression)

def merge_experiment_results(
    config: dict,
    output_dir: Path,
    relative_file_path: str,
    output_filename: str,
    description: str,
    warning_label: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Generic function to merge CSV files from multiple experiments.

    Parameters
    ----------
    config : dict
        Configuration containing the list of experiments.
    output_dir : Path
        Directory where the merged output should be saved.
    relative_file_path : str
        Relative path to the CSV file inside each experiment directory.
    output_filename : str
        Name of the output merged CSV file.
    description : str
        Description used for tqdm and print messages.
    warning_label : str, optional
        Label used in warning messages to specify which file type is missing.
    
    Returns
    -------
    pd.DataFrame or None
        The merged DataFrame if successful, otherwise None.
    """
    all_dfs = []
    pbar = tqdm(config['estimators'], desc=f"Merging {description}")

    for estimator, experiment in config['estimators'].items():
        exp_name = estimator.upper()
        results_path = expand_env_variables_in_path(experiment['experiment-dir'])
        file_path = results_path / relative_file_path

        if not file_path.exists():
            label = warning_label or description
            print(f"\nWARNING: {label} file not found for {exp_name} at {file_path}")
            pbar.update(1)
            continue

        df = pd.read_csv(file_path)
        header_row = pd.DataFrame(
            [[f"--- {exp_name} ---"] + [None] * (len(df.columns) - 1)],
            columns=df.columns
        )
        df = pd.concat([header_row, df], ignore_index=True)

        all_dfs.append(df)
        pbar.update(1)

    pbar.close()

    if not all_dfs:
        print(f"No valid files found for {description}. Skipping merge.")
        return None

    merged_df = pd.concat(all_dfs, ignore_index=True)
    output_file = output_dir / output_filename
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {description} saved to {output_file}")
    print("Done!", flush=True)

    return merged_df


# ---- Specific helper wrappers ---- #
def merge_regression_metrics_dap(config: dict, output_dir: Path) -> Optional[pd.DataFrame]:
    return merge_experiment_results(
        config=config,
        output_dir=output_dir,
        relative_file_path="evaluation/regression/regression_metrics_dap.csv",
        output_filename="regression_metrics_dap.csv",
        description="regression metrics DAP"
    )


def merge_regression_metrics_all_samples(config: dict, output_dir: Path) -> Optional[pd.DataFrame]:
    return merge_experiment_results(
        config=config,
        output_dir=output_dir,
        relative_file_path="evaluation/regression/regression_metrics_all_samples.csv",
        output_filename="regression_metrics_all_samples.csv",
        description="regression metrics all samples"
    )


def merge_calibration_metrics_summary(config: dict, output_dir: Path) -> Optional[pd.DataFrame]:
    return merge_experiment_results(
        config=config,
        output_dir=output_dir,
        relative_file_path="evaluation/calibration/calibration_metrics_summary.csv",
        output_filename="calibration_metrics_summary.csv",
        description="calibration metrics summary"
    )


def merge_coverage_per_target(config: dict, output_dir: Path) -> Optional[pd.DataFrame]:
    return merge_experiment_results(
        config=config,
        output_dir=output_dir,
        relative_file_path="evaluation/coverage/coverage_per_target.csv",
        output_filename="coverage_per_target.csv",
        description="coverage per target"
    )


def merge_log_probs_true_thetas(config: dict, output_dir: Path) -> Optional[pd.DataFrame]:
    return merge_experiment_results(
        config=config,
        output_dir=output_dir,
        relative_file_path="evaluation/log_probs/log_probs_very_short_summary.csv",
        output_filename="log_probs_very_short_summary.csv",
        description="log probs true thetas"
    )



if __name__ == "__main__":
    script_start = time()
    print("\nRUN COMPARISON\n")

    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to the configuration file specifying the estimators to \
            compare."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where to save the merged comparison results.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = safe_load(f)

    output_dir = expand_env_variables_in_path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Stage 1: Load regression results and merge into a single dataframe
    # -------------------------------------------------
    print(80 * "-", flush=True)
    print("(1) Merge regression results into a single dataframe", flush=True)
    print(80 * "-" + "\n", flush=True)
    
    merged_df = merge_regression_metrics_dap(
        config, 
        output_dir
    )

    merged_df = merge_regression_metrics_all_samples(
        config, 
        output_dir
    )

    print("Done!\n\n")

    # -------------------------------------------------
    # Stage 2: Load calibration results and merge into a single dataframe
    # -------------------------------------------------
    print(80 * "-", flush=True)
    print("(2) Merge calibration results into a single dataframe", flush=True)
    print(80 * "-" + "\n", flush=True)
    
    merged_df = merge_calibration_metrics_summary(
        config, 
        output_dir
    )
    print("Done!\n\n")

    # -------------------------------------------------
    # Stage 3: Load coverage results and merge into a single dataframe
    # -------------------------------------------------
    print(80 * "-", flush=True)
    print("(3) Merge coverage results into a single dataframe", flush=True)
    print(80 * "-" + "\n", flush=True)

    merged_df = merge_coverage_per_target(
        config, 
        output_dir
    )

    print("Done!\n\n")

    # -------------------------------------------------
    # Stage 4: Load log probs results and merge into a single dataframe
    # -------------------------------------------------
    print(80 * "-", flush=True)
    print("(4) Merge log probs results into a single dataframe", flush=True)
    print(80 * "-" + "\n", flush=True)
    merged_df = merge_log_probs_true_thetas(
        config, 
        output_dir
    )
    print("Done!\n\n")


    # Print the total runtime
    print(f"This took {time() - script_start:.1f} seconds!\n")
