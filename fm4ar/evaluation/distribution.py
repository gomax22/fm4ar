# Inspired by fm4ar/scripts/importance_sampling/aggregate_test_set_runs.py

import torch
import numpy as np
import pandas as pd

from typing import List
from pathlib import Path

from scipy.spatial.distance import jensenshannon
from fm4ar.utils.distributions import compute_smoothed_histogram
from scipy.stats import wasserstein_distance_nd
from ignite.metrics import MaximumMeanDiscrepancy



# We can compute the Jensen-Shannon divergence between two distributions P and Q
# where P is the marginal distribution of the atmospheric parameters (thetas) 
# and Q is the marginal distribution of the posterior samples obtained after
# sampling. 
# We have two options:
# 1) Aggregate all posterior samples across all observations and compute the
#    JSD between the aggregated posterior and the marginal data distribution.
# 2) Compute the JSD for each observation separately and then average the JSDs.
# We implement both options here.
# TODO: use importance weights when computing the JSD
def measure_jensen_shannon_divergence(
    thetas: np.ndarray,
    posterior_samples: np.ndarray,
) -> np.ndarray:
    """
    Measure the Jensen-Shannon divergence (JSD) between the marginal data
    distributions and marginal posterior distributions.
    Args:
        thetas: Array of shape (n_samples, dim_theta) holding the model
            parameters.
        posterior_samples: Array of shape (n_samples, n_repeats, dim_theta)
            holding the posterior samples.
    Returns:
        jsds: Array of shape (dim_theta,) holding the JSDs in mnat.

    """
    num_samples, num_repeats, dim_theta = posterior_samples.shape
    results = {
        "aggregate": np.full(dim_theta, np.nan),
        "per_observation": np.full((num_repeats, dim_theta), np.nan),
    }

    # Aggregate all posterior samples across all observations
    jsds = np.full(dim_theta, np.nan)
    for i in range(dim_theta):

        # Construct the bins
        # We use the full prior range here because otherwise the results
        # might not be properly comparable between different runs
        bins = np.linspace(thetas[:, i].min(), thetas[:, i].max(), 101)

        _, hist_thetas = compute_smoothed_histogram(
            bins=bins,
            samples=thetas[:, i],
            weights=None,
            sigma=3,  # note: smoothing sigma!
        )
        _, hist_posterior = compute_smoothed_histogram(
            bins=bins,
            samples=posterior_samples[:, :, i].flatten(),
            weights=None,
            sigma=3,  # note: smoothing sigma!
        )
        jsds[i] = 1000 * jensenshannon(hist_thetas, hist_posterior)

    results["aggregate"] = jsds

    # Compute the JSD for each observation separately
    jsds = np.full((num_repeats, dim_theta), np.nan)
    for i in range(dim_theta):
        # Construct the bins
        bins = np.linspace(thetas[:, i].min(), thetas[:, i].max(), 101)
        
        for j in range(num_repeats):
            _, hist_thetas = compute_smoothed_histogram(
                bins=bins,
                samples=thetas[:, i],
                weights=None,
                sigma=3,  # note: smoothing sigma!
            )
            _, hist_posterior = compute_smoothed_histogram(
                bins=bins,
                samples=posterior_samples[:, j, i],
                weights=None,
                sigma=3,  # note: smoothing sigma!
            )
            jsds[j, i] = 1000 * jensenshannon(hist_thetas, hist_posterior)
    results["per_observation"] = jsds

    return results


def save_jsds_to_csv(
    metrics: dict[str, np.ndarray],
    output_dir: Path,
    labels: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save JSD metrics to a CSV file.
    Args:
        metrics: Dictionary containing JSD metrics.
        output_dir: Directory where the CSV file will be saved.
        labels: List of parameter names corresponding to the dimensions.

    Returns:
        df_aggregate: DataFrame containing aggregate JSD metrics.
        df_per_observation: DataFrame containing per-observation JSD metrics.
    """

    # Ensure the output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create a DataFrame to hold the JSD metrics
    df_aggregate = pd.DataFrame({
        "parameters": labels, 
        "jsd_aggregate": metrics["aggregate"]
    })
    df_aggregate.to_csv(
        output_dir / "jsd_aggregate_summary.csv", 
        index=False,
    )
    df_per_observation = pd.DataFrame(
        {"parameters": labels, 
         "means": np.mean(metrics["per_observation"], axis=0),
         "stds": np.std(metrics["per_observation"], axis=0),
        "mins": np.min(metrics["per_observation"], axis=0),
        "maxs": np.max(metrics["per_observation"], axis=0),
        }
    )
    df_per_observation.to_csv(
        output_dir / "jsd_per_observation_summary.csv", 
        index=False,
    )
    return df_aggregate, df_per_observation
    

# We can compute the MMD between two distributions P and Q.
# where P is the marginal distribution of the atmospheric parameters (thetas) 
# and Q is the marginal distribution of the posterior samples obtained after
# sampling. 
def compute_maximum_mean_discrepancy(
    thetas: np.ndarray,
    posterior_samples: np.ndarray,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Compute the Maximum Mean Discrepancy (MMD) between the posterior samples
    and the true parameters.
    Args:
        thetas: Array of shape (n_samples, dim_theta) holding the model
            parameters.
        posterior_samples: Array of shape (n_samples, n_repeats, dim_theta)
            holding the posterior samples.
        device: Device to use for computations.

    Returns:
        results: Dictionary containing MMD metrics.
    """
    # Aggregate all posterior samples across all observations
    n_samples, n_repeats, dim_theta = posterior_samples.shape

    # results = {
    #     "aggregate": np.full(dim_theta, np.nan),
    #     "per_observation": np.full((n_repeats, dim_theta), np.nan),
    # }

    # metric = MaximumMeanDiscrepancy()

    # metric.update(
    #     (torch.tensor(posterior_samples.reshape(-1, dim_theta), dtype=torch.float32),
    #      torch.tensor(thetas.repeat(n_repeats, axis=0), dtype=torch.float32))
    # )
    # mmd_value = metric.compute()
    # results["aggregate"] = mmd_value

    # Compute the MMD for each observation separately
    mmd_values = np.full((n_repeats, dim_theta), np.nan)
    for i in range(n_repeats):
        metric = MaximumMeanDiscrepancy(
            device=device
        )
        metric.update(
            (torch.tensor(posterior_samples[:, i, :], dtype=torch.float32, device=device),
             torch.tensor(thetas, dtype=torch.float32, device=device))
        )
        mmd_values[i] = metric.compute()
    # results["per_observation"] = mmd_values
    return {
        "per_observation": mmd_values
    }

def save_mmd_to_csv(
    metrics: dict[str, float],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save MMD metrics to CSV files.
    Args:
        metrics: Dictionary containing MMD metrics.
        output_dir: Directory where the CSV files will be saved.
        labels: List of labels for the metrics.
    Returns:
        aggregate_df: DataFrame containing aggregate MMD metrics.
        per_observation_df: DataFrame containing per-observation MMD metrics.   
    """

    # Ensure the output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create a DataFrame to hold the Wasserstein distances
    df = pd.DataFrame({
        # "mmd_aggregate": [metrics["aggregate"]],
        "mmd_per_observation_mean": [np.mean(metrics["per_observation"])],
        "mmd_per_observation_std": [np.std(metrics["per_observation"])],
        "mmd_per_observation_min": [np.min(metrics["per_observation"])],
        "mmd_per_observation_max": [np.max(metrics["per_observation"])],
    })

    # Downcast numeric columns to save space
    df = df.apply(pd.to_numeric, downcast='float')

    # Save the DataFrame to a CSV file
    df.to_csv(
        output_dir / "mmd_summary.csv", 
        index=False,
    )
    return df


# We can compute the Wasserstein distance between two distributions P and Q
# where P is the marginal distribution of the atmospheric parameters (thetas)
# and Q is the marginal distribution of the posterior samples obtained after
# sampling.
# TODO: use importance weights when computing the Wasserstein distance
def compute_wasserstein_distance_nd(
    thetas: np.ndarray,
    posterior_samples: np.ndarray,
) -> dict[str, np.ndarray]: 
    """
    Compute the Wasserstein distance between the posterior samples
    and the true parameters.
    Args:
        thetas: Array of shape (n_samples, dim_theta) holding the model
            parameters.
        posterior_samples: Array of shape (n_samples, n_repeats, dim_theta)
            holding the posterior samples.
    Returns:
        results: Dictionary containing Wasserstein distance metrics.
    """
    
    # Resample posterior samples according to the weights
    n_samples, n_repeats, dim_theta = posterior_samples.shape

    # Aggregate all posterior samples across all observations
    # wd_aggregate = wasserstein_distance_nd(
    #     thetas,
    #     posterior_samples.reshape(-1, thetas.shape[1])
    # )
    # results["aggregate"] = wd_aggregate


    # Compute the Wasserstein distance for each observation separately
    wd_per_observation = np.full(n_repeats, np.nan)
    for i in range(n_repeats):
        wd_per_observation[i] = wasserstein_distance_nd(
            thetas,
            posterior_samples[:, i, :]
        )
    
    return {
        "per_observation": wd_per_observation,
    }

def save_wasserstein_distances_to_csv(
    metrics: dict,
    output_dir: Path,
) -> pd.DataFrame:
    
    """
    Save Wasserstein distances to a CSV file.

    Args:
        metrics: Dictionary containing Wasserstein distances.
        output_dir: Directory where the CSV file will be saved.
    """

    # Ensure the output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


    # Create a DataFrame to hold the Wasserstein distances
    df = pd.DataFrame({
        # "wd_aggregate": [metrics["aggregate"]],
        "wd_per_observation_mean": [np.mean(metrics["per_observation"])],
        "wd_per_observation_std": [np.std(metrics["per_observation"])],
        "wd_per_observation_min": [np.min(metrics["per_observation"])],
        "wd_per_observation_max": [np.max(metrics["per_observation"])],
    })

    # Downcast numeric columns to save space
    df = df.apply(pd.to_numeric, downcast='float')

    # Save the DataFrame to a CSV file
    df.to_csv(
        output_dir / "wasserstein_distances_summary.csv", 
        index=False,
    )
    return df


def compute_distribution_metrics(
    thetas: np.ndarray,
    posterior_samples: np.ndarray,
    device: str = "cpu",
) -> dict[str, dict]:
    """
    Compute distribution metrics between the true parameters and posterior samples.
    Args:
        thetas: Array of shape (n_samples, dim_theta) holding the model
            parameters.
        posterior_samples: Array of shape (n_samples, n_repeats, dim_theta)
            holding the posterior samples.
        device: Device to use for computations.

    Returns:
        metrics: Dictionary containing distribution metrics.
    """

    metrics = {}
    print("Computing JSD metrics...", end=" ", flush=True)
    jsd_metrics = measure_jensen_shannon_divergence(
        thetas=thetas,
        posterior_samples=posterior_samples,
    )
    metrics["jsd"] = jsd_metrics
    print("Done!")

    print("Computing MMD metrics...", end=" ", flush=True)
    mmd_metrics = compute_maximum_mean_discrepancy(
        thetas=thetas,
        posterior_samples=posterior_samples,
        device=device,
    )
    metrics["mmd"] = mmd_metrics
    print("Done!")

    # print("Computing Wasserstein distance metrics...", end=" ", flush=True)
    # wd_metrics = compute_wasserstein_distance_nd(
    #     thetas=thetas,
    #     posterior_samples=posterior_samples,
    # )
    # metrics["wasserstein_distance"] = wd_metrics
    # print("Done!")

    return metrics

def save_distribution_metrics_to_csv(
    metrics: dict[str, dict],
    output_dir: Path,
    labels: List[str],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Save distribution metrics to CSV files.
    Args:
        metrics: Dictionary containing distribution metrics.
        output_dir: Directory where the CSV files will be saved.
        labels: List of parameter names corresponding to the dimensions.
    Returns:
        saved_dfs: Dictionary containing DataFrames of saved metrics.
    """
    saved_dfs = {}

    jsd_dfs = save_jsds_to_csv(
        metrics=metrics["jsd"],
        output_dir=output_dir,
        labels=labels,
    )
    saved_dfs["jsd"] = jsd_dfs

    mmd_df = save_mmd_to_csv(
        metrics=metrics["mmd"],
        output_dir=output_dir,
    )
    saved_dfs["mmd"] = mmd_df

    # wd_df = save_wasserstein_distances_to_csv(
    #     metrics=metrics["wasserstein_distance"],
    #     output_dir=output_dir,
    # )
    # saved_dfs["wasserstein_distance"] = wd_df

    return saved_dfs