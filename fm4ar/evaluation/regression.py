"""
Utilities to compute regression metrics for posterior distributions
"""

import torch
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    root_mean_squared_error,
)

def make_violin_plot(
    metrics: dict,
    output_fname: Path,
    labels: List[str],
    figsize: tuple[int, int] = (8, 16),
    fontsize: int = 14,
    y_scale: str = 'log',
) -> None:
    
    # Extract number of targets
    metrics_keys = list(metrics.keys())
    n_targets = metrics[metrics_keys[0]].shape[1]

    # Make a violin plot for each target (row) and each metric (column)
    fig, axs = plt.subplots(nrows=len(labels), ncols=1, figsize=figsize)
    for target_idx in range(n_targets):
        plots = axs[target_idx].violinplot(
            [errors[:, target_idx] for _, errors in metrics.items()], 
            showmeans=True, 
            showmedians=True, 
            showextrema=True
        )

        # Make all the violin statistics marks red:
        # for pc, color in zip(plots['bodies'], colors):
        # for pc in plots['bodies']:
        #     pc.set_facecolor('red')
        #     pc.set_edgecolor('black')

        # # Set the color of the median lines
        # for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        #     plots[partname].set_colors(colors)
        
        axs[target_idx].set_ylabel(labels[target_idx], fontsize=fontsize)
        axs[target_idx].set_xticks(range(1, len(list(metrics.keys())) + 1))
        axs[target_idx].set_xticklabels(list(metrics.keys()), fontsize=fontsize)
        axs[target_idx].set_yscale(y_scale)
        axs[target_idx].grid()

    fig.tight_layout()
    fig.savefig(f"{output_fname}.png", dpi=400)
    fig.savefig(f"{output_fname}.pdf", format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)

    return None


# TODO: make violin plots of errors per target, per sample, for each metric (mse, mae, rmse, median ae)
def make_violin_plots_of_errors(
    errors: dict,
    output_dir: Path,
    labels: List[str],
    figsize: tuple[int, int] = (8, 16),
    fontsize: int = 14,
    y_scale: str = 'log',
):
    
    # Extract squared and absolute errors
    sqrd_errors = errors['sqrd_errors'] # (n_samples, n_repeats, n_targets)
    abs_errors = errors['abs_errors'] # (n_samples, n_repeats, n_targets)

    metrics = {
        'MSE': torch.mean(sqrd_errors, dim=1).numpy(),
        'MAE': torch.mean(abs_errors, dim=1).numpy(),
        'RMSE': torch.sqrt(torch.mean(sqrd_errors, dim=1)).numpy(),
        'Median AE': torch.median(abs_errors, dim=1).values.numpy()
    }

    # Make a violin plot for each target (row) and each metric (column)
    make_violin_plot(
        metrics=metrics,
        output_fname=Path(output_dir) / "error_distributions",
        labels=labels,
        figsize=figsize,
        fontsize=fontsize,
        y_scale=y_scale 
    )

    # Extract DAP errors
    dap_sqrd_errors = errors['dap_sqrd_errors'] # (n_samples, n_targets)
    dap_abs_errors = errors['dap_abs_errors'] # (n_samples, n_targets)
    
    # If DAP errors are provided, make violin plots for them too
    if dap_sqrd_errors is not None and dap_abs_errors is not None:
        dap_metrics = {
            'MSE': torch.mean(dap_sqrd_errors, dim=0, keepdim=True).numpy(),
            'MAE': torch.mean(dap_abs_errors, dim=0, keepdim=True).numpy(),
            'RMSE': torch.sqrt(torch.mean(dap_sqrd_errors, dim=0, keepdim=True)).numpy(),
            'Median AE': torch.median(dap_abs_errors, dim=0, keepdim=True).values.numpy()
        }
        
        make_violin_plot(
            metrics=dap_metrics,
            output_fname=Path(output_dir) / "dap_error_distributions",
            labels=labels,
            figsize=figsize,
            fontsize=fontsize,
            y_scale=y_scale 
        )


def compute_regression_metrics_using_scikit_learn(
    thetas:  np.ndarray,
    posterior_samples: np.ndarray,
    dap_samples: np.ndarray | None = None,
) -> None:
    
    num_samples, num_repeats, _ = posterior_samples.shape
    mae_s, mse_s, rmse_s, medae_s = [], [], [], []

    for s in range(num_samples):
        yp = posterior_samples[s, :, :]
        yt = thetas[s, :][np.newaxis, :].repeat(num_repeats, axis=0)
        mae_s.append(mean_absolute_error(yt, yp, multioutput='raw_values'))
        mse_s.append(mean_squared_error(yt, yp, multioutput='raw_values'))
        rmse_s.append(root_mean_squared_error(yt, yp, multioutput='raw_values'))
        medae_s.append(median_absolute_error(yt, yp, multioutput='raw_values'))

    # Stack → shape: (num_samples, num_targets)
    mae_s   = np.stack(mae_s)
    mse_s   = np.stack(mse_s)
    rmse_s  = np.stack(rmse_s)
    medae_s = np.stack(medae_s)


    dist_metrics = {
        'per_target': {
            'mse': {
                'mean': mse_s.mean(0),
                'std':  mse_s.std(0),
                'min': mse_s.min(0),
                'max': mse_s.max(0),
                'median': np.median(mse_s, axis=0),
            },
            'mae': {
                'mean': mae_s.mean(0),
                'std':  mae_s.std(0),
                'min': mae_s.min(0),
                'max': mae_s.max(0),
                'median': np.median(mae_s, axis=0),
            },
            'rmse': {
                'mean': rmse_s.mean(0),
                'std':  rmse_s.std(0),
                'min': rmse_s.min(0),
                'max': rmse_s.max(0),
                'median': np.median(rmse_s, axis=0),
            },
            'median_ae': {
                'mean': medae_s.mean(0),
                'std':  medae_s.std(0),
                'min': medae_s.min(0),
                'max': medae_s.max(0),
                'median': np.median(medae_s, axis=0),
            },
        },
        'per_sample': {
            'mse': {
                'mean': mse_s.mean(1),
                'std':  mse_s.std(1),
                'min': mse_s.min(1),
                'max': mse_s.max(1),
                'median': np.median(mse_s, axis=1),
            },
            'mae': {
                'mean': mae_s.mean(1),
                'std':  mae_s.std(1),
                'min': mae_s.min(1),
                'max': mae_s.max(1),
                'median': np.median(mae_s, axis=1),
            },
            'rmse': {
                'mean': rmse_s.mean(1),
                'std':  rmse_s.std(1),
                'min': rmse_s.min(1),
                'max': rmse_s.max(1),
                'median': np.median(rmse_s, axis=1),
            },
            'median_ae': {
                'mean': medae_s.mean(1),
                'std':  medae_s.std(1),
                'min': medae_s.min(1),
                'max': medae_s.max(1),
                'median': np.median(medae_s, axis=1),
            },
        }
    }

    dap_metrics = None
    if dap_samples is not None:
            
        # Now do it for DAP samples
        mae_s, mse_s, rmse_s, medae_s = [], [], [], []

        for s in range(num_samples):
            yp = dap_samples[s, :][np.newaxis, :]
            yt = thetas[s, :][np.newaxis, :]
            mae_s.append(mean_absolute_error(yt, yp, multioutput='raw_values'))
            mse_s.append(mean_squared_error(yt, yp, multioutput='raw_values'))
            rmse_s.append(root_mean_squared_error(yt, yp, multioutput='raw_values'))
            medae_s.append(median_absolute_error(yt, yp, multioutput='raw_values'))

        # Stack → shape: (num_samples, num_targets)
        mae_s   = np.stack(mae_s)
        mse_s   = np.stack(mse_s)
        rmse_s  = np.stack(rmse_s)
        medae_s = np.stack(medae_s)

        dap_metrics = {
            'mse': {
                'mean': mse_s.mean(0),
                'std':  mse_s.std(0),
                'min': mse_s.min(0),
                'max': mse_s.max(0),
                'median': np.median(mse_s, axis=0),
            },
            'mae': {
                'mean': mae_s.mean(0),
                'std':  mae_s.std(0),
                'min': mae_s.min(0),
                'max': mae_s.max(0),
                'median': np.median(mae_s, axis=0),
            },
            'rmse': {
                'mean': rmse_s.mean(0),
                'std':  rmse_s.std(0),
                'min': rmse_s.min(0),
                'max': rmse_s.max(0),
                'median': np.median(rmse_s, axis=0),
            },
            'median_ae': {
                'mean': medae_s.mean(0),
                'std':  medae_s.std(0),
                'min': medae_s.min(0),
                'max': medae_s.max(0),
                'median': np.median(medae_s, axis=0),
            },
        }

    return {
        'distribution_metrics': dist_metrics,
        'dap_metrics': dap_metrics
    }

    
def compute_regression_metrics_from_errors(
    sqrd_errors: torch.Tensor,
    abs_errors: torch.Tensor,
    dap_sqrd_errors: torch.Tensor | None = None,
    dap_abs_errors: torch.Tensor | None = None,
) -> dict:
    """Compute mse, mae, rmse, median absolute errors from squared and absolute errors.

    Args:
        sqrd_errors: Squared errors tensor of shape (n_samples, n_repeats, n_targets).
        abs_errors: Absolute errors tensor of shape (n_samples, n_repeats, n_targets).
        dap_sqrd_errors: Squared errors for DAP samples of shape (n_samples, n_targets).
        dap_abs_errors: Absolute errors for DAP samples of shape (n_samples, n_targets).

    Returns:
        A dictionary containing computed metrics. If DAP errors are provided,
        metrics for DAP samples are included.
    """

    # Compute metrics on the entire posterior distribution
    dist_metrics = {
        'per_target': {
            'mse': {
                'mean': torch.mean(torch.mean(sqrd_errors, dim=1), dim=0).numpy(),
                'std': torch.std(torch.mean(sqrd_errors, dim=1), dim=0).numpy(),
                'min': torch.min(torch.mean(sqrd_errors, dim=1), dim=0).values.numpy(),
                'max': torch.max(torch.mean(sqrd_errors, dim=1), dim=0).values.numpy(),
                'median': torch.median(torch.mean(sqrd_errors, dim=1), dim=0).values.numpy()
            },
            'mae': {
                'mean' : torch.mean(torch.mean(abs_errors, dim=1), dim=0).numpy(),
                'std': torch.std(torch.mean(abs_errors, dim=1), dim=0).numpy(),
                'min': torch.min(torch.mean(abs_errors, dim=1), dim=0).values.numpy(),
                'max': torch.max(torch.mean(abs_errors, dim=1), dim=0).values.numpy(),
                'median': torch.median(torch.mean(abs_errors, dim=1), dim=0).values.numpy()
            },
            'rmse': {
                'mean': torch.mean(torch.mean(sqrd_errors, dim=1).sqrt(), dim=0).numpy(),
                'std': torch.std(torch.mean(sqrd_errors, dim=1).sqrt(), dim=0).numpy(),
                'min': torch.min(torch.mean(sqrd_errors, dim=1).sqrt(), dim=0).values.numpy(),
                'max': torch.max(torch.mean(sqrd_errors, dim=1).sqrt(), dim=0).values.numpy(),
                'median': torch.median(torch.mean(sqrd_errors, dim=1).sqrt(), dim=0).values.numpy()
            },
            'median_ae': {
                'mean': torch.mean(abs_errors.median(dim=1).values, dim=0).numpy(),
                'std': torch.std(abs_errors.median(dim=1).values, dim=0).numpy(),
                'min': torch.min(abs_errors.median(dim=1).values, dim=0).values.numpy(),
                'max': torch.max(abs_errors.median(dim=1).values, dim=0).values.numpy(),
                'median': torch.median(abs_errors.median(dim=1).values, dim=0).values.numpy()
            }
        },
        'per_sample': {
            'mse': {
                'mean': torch.mean(torch.mean(sqrd_errors, dim=1), dim=1).numpy(),
                'std': torch.std(torch.mean(sqrd_errors, dim=1), dim=1).numpy(),
                'min': torch.min(torch.mean(sqrd_errors, dim=1), dim=1).values.numpy(),
                'max': torch.max(torch.mean(sqrd_errors, dim=1), dim=1).values.numpy(),
                'median': torch.median(torch.mean(sqrd_errors, dim=1), dim=1).values.numpy()
            },
            'mae': {
                'mean' : torch.mean(torch.mean(abs_errors, dim=1), dim=1).numpy(),
                'std': torch.std(torch.mean(abs_errors, dim=1), dim=1).numpy(),
                'min': torch.min(torch.mean(abs_errors, dim=1), dim=1).values.numpy(),
                'max': torch.max(torch.mean(abs_errors, dim=1), dim=1).values.numpy(),
                'median': torch.median(torch.mean(abs_errors, dim=1), dim=1).values.numpy()
            },
            'rmse': {
                'mean': torch.mean(torch.mean(sqrd_errors, dim=1).sqrt(), dim=1).numpy(),
                'std': torch.std(torch.mean(sqrd_errors, dim=1).sqrt(), dim=1).numpy(),
                'min': torch.min(torch.mean(sqrd_errors, dim=1).sqrt(), dim=1).values.numpy(),
                'max': torch.max(torch.mean(sqrd_errors, dim=1).sqrt(), dim=1).values.numpy(),
                'median': torch.median(torch.mean(sqrd_errors, dim=1).sqrt(), dim=1).values.numpy()
            },
            'median_ae': {
                'mean': torch.mean(abs_errors.median(dim=1).values, dim=1).numpy(),
                'std': torch.std(abs_errors.median(dim=1).values, dim=1).numpy(),
                'min': torch.min(abs_errors.median(dim=1).values, dim=1).values.numpy(),
                'max': torch.max(abs_errors.median(dim=1).values, dim=1).values.numpy(),
                'median': torch.median(abs_errors.median(dim=1).values, dim=1).values.numpy()
            }
        },
    }
    dap_metrics = None
    if dap_sqrd_errors is not None and dap_abs_errors is not None:
        # Compute metrics on the DAP samples
        dap_metrics = {
            'mse': {
                'mean': torch.mean(dap_sqrd_errors, dim=0).numpy(),
                'std': torch.std(dap_sqrd_errors, dim=0).numpy(),
                'min': torch.min(dap_sqrd_errors, dim=0).values.numpy(),
                'max': torch.max(dap_sqrd_errors, dim=0).values.numpy(),
                'median': torch.median(dap_sqrd_errors, dim=0).values.numpy()
            },
            'mae': {
                'mean': torch.mean(dap_abs_errors, dim=0).numpy(),
                'std': torch.std(dap_abs_errors, dim=0).numpy(),
                'min': torch.min(dap_abs_errors, dim=0).values.numpy(),
                'max': torch.max(dap_abs_errors, dim=0).values.numpy(),
                'median': torch.median(dap_abs_errors, dim=0).values.numpy()
            },
            'rmse': {
                'mean': torch.mean(dap_sqrd_errors, dim=0).sqrt().numpy(),
                'std': torch.std(dap_sqrd_errors, dim=0).sqrt().numpy(),
                'min': torch.min(dap_sqrd_errors, dim=0).values.sqrt().numpy(),
                'max': torch.max(dap_sqrd_errors, dim=0).values.sqrt().numpy(),
                'median': torch.median(dap_sqrd_errors, dim=0).values.sqrt().numpy()

            },
            'median_ae': {
                'mean': torch.median(dap_abs_errors, dim=0).values.numpy(),
                'std': torch.std(dap_abs_errors, dim=0).numpy(),
                'min': torch.min(dap_abs_errors, dim=0).values.numpy(),
                'max': torch.max(dap_abs_errors, dim=0).values.numpy(),
                'median': torch.median(dap_abs_errors, dim=0).values.numpy()
            }
        }
    return {
        'distribution_metrics': dist_metrics,
        'dap_metrics': dap_metrics
    }

def calculate_errors(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    dap_samples: np.ndarray | None = None,
) -> dict[str, torch.Tensor | None]:
    """
    Calculate squared and absolute errors between posterior samples and true parameters.
    """
    
    # Compute mse, mae, rmse, median absolute errors
    _, n_repeats, _ = posterior_samples.shape
    thetas = np.expand_dims(thetas, axis=1).repeat(n_repeats, axis=1)
    
    sqrd_errors = F.mse_loss(
        torch.from_numpy(posterior_samples),
        torch.from_numpy(thetas),
        reduction='none'
    )
    abs_errors = F.l1_loss(
        torch.from_numpy(posterior_samples),
        torch.from_numpy(thetas),
        reduction='none'
    )

    dap_sqrd_errors = None
    dap_abs_errors = None
    if dap_samples is not None:
        # Compute metrics on the DAP samples
        dap_sqrd_errors = F.mse_loss(
            torch.from_numpy(dap_samples),
            torch.from_numpy(thetas[:, 0, :]),
            reduction='none'
        )
        dap_abs_errors = F.l1_loss(
            torch.from_numpy(dap_samples),
            torch.from_numpy(thetas[:, 0, :]),
            reduction='none'
        )

    return {
        'sqrd_errors': sqrd_errors,
        'abs_errors': abs_errors,
        'dap_sqrd_errors': dap_sqrd_errors,
        'dap_abs_errors': dap_abs_errors
    }


def save_metrics_to_csv(
    metrics: dict,
    output_dir: Path,
    labels: List[str],
    dap: bool = False,
    prefix: str = ''
) -> None:
    """
    Utility to save regression metrics as csv files.
    Args:
        metrics: A dictionary containing regression metrics.
        output_dir: Directory to save the csv files.
        labels: List of target labels.
        dap: Whether the metrics correspond to DAP samples.
        prefix: Optional prefix for the output filenames.

    """
    

    # Extract metrics and stats keys
    metrics_keys = list(metrics.keys())
    stats = list(metrics[metrics_keys[0]].keys())

    # flatten metrics into a single row
    # columns = [f"{l}_{m}_{s}" for l in labels for m in metrics_keys for s in stats]
    columns = ["parameters"] + [f"{m}_{s}" for m in metrics_keys for s in stats]
    data = np.array([
        [l] + [metrics[m][s][i].item() for m in metrics_keys for s in stats] 
        for i, l in enumerate(labels)
    ]).squeeze()

    posterior_samples_type = 'dap' if dap else 'all_samples'

    # Create dataframe with targets on rows and metrics on columns
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_dir / f'{prefix}regression_metrics_{posterior_samples_type}.csv', index=False)

    # Create and save a dataframe for each metric
    for metric_name in metrics_keys:

        # flatten metrics into a single row
        columns = ["parameters"] + [f"{metric_name}_{s}" for s in stats]
        data = np.array([
            [l] + [metrics[metric_name][s][i].item() for s in stats] 
            for i, l in enumerate(labels)
        ]).squeeze()

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_dir / f'{prefix}regression_{metric_name}_{posterior_samples_type}.csv', index=False)


def save_regression_errors_to_csv(
    sqrd_errors: torch.Tensor,
    abs_errors: torch.Tensor,
    dap_sqrd_errors: torch.Tensor | None,
    dap_abs_errors: torch.Tensor | None,
    output_dir: Path,
    labels: List[str],
) -> None:

    """
    Utility to save regression errors as csv files.
    Args:
        sqrd_errors: Squared errors tensor of shape (n_samples, n_repeats, n_targets).
        abs_errors: Absolute errors tensor of shape (n_samples, n_repeats, n_targets).
        dap_sqrd_errors: Squared errors for DAP samples of shape (n_samples, n_targets).
        dap_abs_errors: Absolute errors for DAP samples of shape (n_samples, n_targets).
        output_dir: Directory to save the csv files.
        labels: List of target labels.
    Returns:
        None
    """
    
    metrics = {
        'mse': torch.mean(sqrd_errors, dim=1).numpy(),
        'mae': torch.mean(abs_errors, dim=1).numpy(),
        'rmse': torch.mean(sqrd_errors, dim=1).sqrt().numpy(),
        'median_ae': torch.median(abs_errors, dim=1).values.numpy(),
    }

    for metric_name, metric_values in metrics.items():
        
        df = pd.DataFrame(metric_values, columns=labels)
        df.to_csv(output_dir / f'regression_{metric_name}_errors_all_samples.csv', index=False)

    if dap_abs_errors is not None and dap_sqrd_errors is not None:
        dap_metrics = {
            'mse': torch.mean(dap_sqrd_errors, dim=0, keepdim=True).numpy(),
            'mae': torch.mean(dap_abs_errors, dim=0, keepdim=True).numpy(),
            'rmse': torch.mean(dap_sqrd_errors, dim=0, keepdim=True).sqrt().numpy(),
            'median_ae': torch.median(dap_abs_errors, dim=0, keepdim=True).values.numpy(),
        }

        indices = list(dap_metrics.keys())
        data = np.concatenate([
                np.array(indices).reshape(-1, 1),
                np.concatenate(
                    [dap_metrics[metric_name] for metric_name in indices],
                    axis=0
                )
            ],
            axis=1
        )
        
        df = pd.DataFrame(data, columns=["metrics"] + labels)
        df.to_csv(output_dir / f'regression_errors_dap.csv', index=False)

    return None


def save_regression_metrics_to_csv(
    metrics: dict,
    output_dir: Path,
    labels: List[str],
    prefix: str = ''
) -> None:
    """
    Save regression metrics to csv files.
    Args:
        metrics: A dictionary containing regression metrics.
        output_dir: Directory to save the csv files.
        labels: List of target labels.
    """
    
    # Save distribution metrics as csv files
    save_metrics_to_csv(
        metrics=metrics['distribution_metrics']['per_sample'],
        output_dir=output_dir,
        labels=labels,
        dap=False,
        prefix=prefix
    )
    
    # If DAP metrics are provided, save them too
    dap_metrics = metrics['dap_metrics']
    if dap_metrics is not None:
        
        save_metrics_to_csv(
            metrics=dap_metrics,
            output_dir=output_dir,
            labels=labels,
            dap=True,
            prefix=prefix
        )

    return None
        
            

