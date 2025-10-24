"""
Utilities to perform coverage analysis for posterior distributions.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import re
from typing import List, Tuple


def initialize_coverage_results(
    statistics: List[str],
    confidence_levels: List[float],
    support: bool
) -> dict:
    """
    Initialize a dictionary to store coverage analysis results.
    Args:
        statistics: List of statistics to compute (e.g., ['posterior_mean', 'posterior_std', 'posterior_accuracy'])
        confidence_levels: List of confidence levels to analyze (e.g., [0.68, 0.95])
        support: bool, whether to include support coverage analysis
    Returns:
        results: A dictionary to store coverage analysis results.
    """
    
    results = {s: [] for s in statistics if s != 'posterior_accuracy'}

    results.update({
        f'posterior_accuracy_{int(cl*100)}': [] for cl in confidence_levels
    })

    if support:
        results.update({
            'posterior_accuracy_support': []
        })
    
    return results

def get_column_labels_per_target_dataframe(
    statistics: List[str],
    confidence_levels: List[float],
    support: bool
) -> List[str]:
    """
    Generate column labels for coverage analysis results.
    Args:
        statistics: List of statistics to compute (e.g., ['posterior_mean', 'posterior_std', 'posterior_accuracy'])
        confidence_levels: List of confidence levels to analyze (e.g., [0.68, 0.95])
        support: bool, whether to include support coverage analysis
    Returns:
        columns_labels: List of column labels for the results dataframe.

        
        """

    columns_label = ["parameters"] + statistics
    for cl in confidence_levels:
        columns_label += [
            f'posterior_accuracy_{int(cl*100)}',
            f'posterior_accuracy_ratio_{int(cl*100)}',
            f'posterior_accuracy_all_{int(cl*100)}',
        ]
    if support:
        columns_label += [
            f'posterior_accuracy_support',
            f'posterior_accuracy_ratio_support',
            f'posterior_accuracy_all_support',
        ]

    return columns_label

def get_column_labels_per_sample_dataframe(
    labels: List[str],
    statistics: List[str],
    confidence_levels: List[float],
    support: bool
) -> List[str]:
    """
    Generate column labels for coverage analysis results.
    Args:
        labels: List of parameter labels.
        statistics: List of statistics to compute (e.g., ['posterior_mean', 'posterior_std', 'posterior_accuracy'])
        confidence_levels: List of confidence levels to analyze (e.g., [0.68, 0.95])
        support: bool, whether to include support coverage analysis
    Returns:
        columns_labels: List of column labels for the results dataframe.

        
        """
    
    columns_label = [
        f'{s}_{l}' for s in statistics for l in labels if s != 'posterior_accuracy'
    ]

    for cl in confidence_levels:
        columns_label += [
            f'posterior_accuracy_{l}_{int(cl*100)}' for l in labels
        ]
        columns_label += [
            f'posterior_accuracy_ratio_{int(cl*100)}',
            f'posterior_accuracy_all_{int(cl*100)}',
        ]
    if support:
        columns_label += [
            f'posterior_accuracy_{l}_support' for l in labels
        ]
        columns_label += [
            f'posterior_accuracy_ratio_support',
            f'posterior_accuracy_all_support',
        ]

    return columns_label


def compute_confidence_interval(
    samples: np.ndarray, 
    confidence: float
):
    """
    Compute confidence interval for samples.

    Args:
        samples: np.ndarray of shape (n_samples, n_dims)
        confidence: float, confidence level (e.g., 0.68 for 1-sigma)
    Returns:
        ci_lower: np.ndarray of shape (n_dims,), lower bound of confidence interval
        ci_upper: np.ndarray of shape (n_dims,), upper bound of confidence interval
    """
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    z_star = stats.norm.interval(confidence)
    ci_lower, ci_upper = mean + z_star[0] * std, mean + z_star[1] * std
    return ci_lower, ci_upper

def analyze_xsigma_coverage(
    posterior_samples: np.ndarray, 
    theta: np.ndarray,
    confidence: float
):
    """
    Compute x-sigma coverage analysis.

    Args:
        posterior_samples: np.ndarray of shape (n_samples, n_dims)
        theta: np.ndarray of shape (n_dims,)
        confidence: float, confidence level (e.g., 0.68 for 1-sigma)
    Returns:
        accuracy: np.ndarray of shape (n_dims,), 1 if theta is within confidence interval, 0 otherwise
    """

    ci_lower, ci_upper = compute_confidence_interval(posterior_samples, confidence)

    # check whether theta is within x std of the posterior mean
    accuracy = np.logical_and(theta >= ci_lower, theta <= ci_upper).astype(np.float32)
    avg_accuracy = np.mean(accuracy, dtype=np.float32).reshape(1,)
    joint_accuracy = np.all(accuracy).astype(np.float32).reshape(1,)
    # print(f"Accuracy one sigma: {accuracy} ({np.mean(accuracy_one_sigma, dtype=np.float32)})")
    return accuracy, avg_accuracy, joint_accuracy, (ci_lower, ci_upper)

def analyze_support_coverage(
    posterior_samples: np.ndarray, 
    thetas: np.ndarray
):  
    """
    Compute support coverage analysis.
    Args:
        posterior_samples: np.ndarray of shape (n_samples, n_dims)
        thetas: np.ndarray of shape (n_dims,)
    Returns:
        accuracy: np.ndarray of shape (n_dims,), 1 if theta is within support, 0 otherwise
    
    """

    # check whether theta[i] is within the support of marginal posterior samples
    
    # WARNING: we assume that there are no holes in the support between max and min
    # POSSIBLE SOLUTION: we need to compute the histogram of the samples for each dimension 
    # and check whether the input theta is within a non-zero bin of the histogram
    # this is a more general solution that can be applied to any distribution (but more expensive)
    # Moreover, the number of bins plays a crucial role in the accuracy of the check
    # and the posteriors having a large variace might require a large number of bins 
    # otherwise it will be unfair 
    # ==> normalization could be beneficial in this case, however we would alter the
    # shape of the distribution and the check would be less meaningful
    min_th = np.min(posterior_samples, axis=0)
    max_th = np.max(posterior_samples, axis=0)
    accuracy = np.logical_and(thetas >= min_th, thetas <= max_th).astype(np.float32)
    avg_accuracy = np.mean(accuracy, dtype=np.float32).reshape(1,)
    joint_accuracy = np.all(accuracy).astype(np.float32).reshape(1,)
    return accuracy, avg_accuracy, joint_accuracy


def analyze_coverage(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    confidence_levels: List[float],
    support: bool
) -> dict:
    """
    Run coverage metrics analysis.
    Args:
        posterior_samples: np.ndarray of shape (n_samples, n_dims)
        thetas: np.ndarray of shape (n_dims,)
        confidence_levels: List of confidence levels to analyze (e.g., [0.68, 0.95])
        support: bool, whether to analyze support coverage
    Returns:
        results: A dictionary containing coverage analysis results.
        
    """
    
    results = {}

    # Analyze support coverage if enabled
    if support:
        res_support = analyze_support_coverage(posterior_samples, thetas)
        results['support'] = {
            "independent_accuracy": res_support[0],
            "avg_accuracy": res_support[1],
            "joint_accuracy": res_support[2]
        }

    # Analyze coverage at different confidence levels
    for cl in confidence_levels:
        res = analyze_xsigma_coverage(posterior_samples, thetas, cl)
        key = f"{int(cl*100)}"
        results[key] = {
            "independent_accuracy": res[0],
            "avg_accuracy": res[1],
            "joint_accuracy": res[2],
            "ci": res[3]
        }
            
    return results


def perform_coverage_analysis(
    posterior_samples: np.ndarray, 
    thetas: np.ndarray,
    labels: List[str],
    output_dir: Path,
    statistics: List[str] = ['posterior_mean', 'posterior_std', 'posterior_accuracy'],
    confidence_levels: List[float] = [0.68, 0.95],
    support: bool = True
) -> pd.DataFrame:
    """
    Perform coverage analysis for posterior distributions.
    Args:
        posterior_samples: np.ndarray of shape (n_samples, n_posterior_samples, n_dims)
        thetas: np.ndarray of shape (n_samples, n_dims)
        labels: List of parameter labels.
        output_dir: Directory to save the results.
        statistics: List of statistics to compute (e.g., ['posterior_mean', 'posterior_std', 'posterior_accuracy'])
        confidence_levels: List of confidence levels to analyze (e.g., [0.68, 0.95])
        support: bool, whether to include support coverage analysis
    Returns:
        coverage_df: pd.DataFrame containing coverage analysis results.
    """

    
    # Ensure output directory exists
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract number of samples
    n_samples = posterior_samples.shape[0]

    # Get column labels
    columns_label = get_column_labels_per_sample_dataframe(
        labels=labels,
        statistics=statistics,
        confidence_levels=confidence_levels,
        support=support
    )

    # Initialize results dictionary
    results = initialize_coverage_results(
        statistics=statistics,
        confidence_levels=confidence_levels,
        support=support
    )

    with tqdm(total=n_samples, desc="Performing posterior coverage analysis...") as pbar:
        for idx, (posterior_samples_batch, theta) in enumerate(zip(posterior_samples, thetas)):
            
            # compute means and stds
            if 'posterior_mean' in statistics:
                results['posterior_mean'].append(np.mean(posterior_samples_batch, axis=0))
            if 'posterior_std' in statistics:
                results['posterior_std'].append(np.std(posterior_samples_batch, axis=0, ddof=1))


            # compute coverages
            coverage_results = analyze_coverage(
                posterior_samples_batch, 
                theta,
                confidence_levels=confidence_levels,
                support=support
            )

            if 'posterior_accuracy' in statistics:
                for cl in confidence_levels:
                    key = f"{int(cl*100)}"
                    results[f'posterior_accuracy_{key}'].append(
                        np.concatenate([
                            coverage_results[key]['independent_accuracy'],
                            coverage_results[key]['avg_accuracy'],
                            coverage_results[key]['joint_accuracy'],
                        ],  axis=0)
                    )
                if support:
                    results['posterior_accuracy_support'].append(
                        np.concatenate([
                            coverage_results['support']['independent_accuracy'],
                            coverage_results['support']['avg_accuracy'],
                            coverage_results['support']['joint_accuracy']
                        ], axis=0)
                    )
            pbar.update(1)
    pbar.close()
    
    coverage_per_sample = np.concatenate([
        v for k, v in results.items()
    ], axis=1)
    
    # create dataframe starting from posterior evaluation
    coverage_per_sample_df = pd.DataFrame(coverage_per_sample, columns=columns_label)    
    coverage_per_sample_df.to_csv(output_dir / 'coverage_per_sample.csv', index=False)


    # create a dataframe with targets on the rows and average over samples
    columns = get_column_labels_per_target_dataframe(
        statistics=statistics,
        confidence_levels=confidence_levels,
        support=support
    )

    # compute average over samples
    data = coverage_per_sample_df.mean(axis=0).to_frame().T

    
    # Extract all relevant columns
    cols = data.columns

    # Initialize list to collect results
    rows = []

    # Define regex to parse column names
    pattern = r"posterior_(mean|std|accuracy)_(.*?)(?:_(\d+|support))?$"

    for col in cols:
        m = re.match(pattern, col)
        if m:
            stat, param, conf = m.groups()
            value = data[col].iloc[0]
            rows.append({
                "parameter": param,
                "stat": stat,
                "confidence": conf if conf else "base",
                "value": value
            })

    tidy = pd.DataFrame(rows)

    # Pivot to desired format
    coverage_per_target_df = (
        tidy
        .pivot_table(index="parameter", columns=["stat", "confidence"], values="value")
        .reset_index()
    )

    # Flatten multi-index columns
    coverage_per_target_df.columns = ['_'.join(filter(None, col)).strip('_') for col in coverage_per_target_df.columns]

    coverage_per_target_df.to_csv(output_dir / 'coverage_per_target.csv', index=False)
    return coverage_per_target_df
