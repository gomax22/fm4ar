import pandas as pd
import numpy as np
from pathlib import Path

def save_log_probs_to_csv(
    log_probs: np.ndarray,
    top_log_probs: np.ndarray,
    log_probs_true_thetas: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    
    """
    Save log probabilities to a CSV file.

    Args:
        log_probs: Array of log probabilities to save (n_samples, n_repeats).
        log_probs_true_theta: Array of log probabilities at the true parameters (n_samples, 1).
        output_dir: Directory where the CSV file will be saved.
    """

    # Ensure the output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)



    # Create a DataFrame to hold the log probabilities
    df = pd.DataFrame(
        np.concatenate(
            [log_probs_true_thetas, top_log_probs, log_probs],
            axis=1,
        ),
        columns=["log_prob_true_thetas", "top_log_prob"] + [f"log_prob_sample_{i+1}" for i in range(log_probs.shape[1])],

    )
    
    # Downcast numeric columns to save space
    df = df.apply(pd.to_numeric, downcast='integer')
    df = df.apply(pd.to_numeric, downcast='float')

    # Make sure the downcasting worked
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype(np.float16)
    
    # Save the DataFrame to a CSV file
    df.to_csv(
        output_dir / "log_probs_long_summary.csv", 
        index=False,
    )

    # Create a DataFrame to hold the log probabilities
    df = pd.DataFrame(
        np.concatenate(
            [
                log_probs_true_thetas, 
                top_log_probs, 
                log_probs.mean(axis=1, keepdims=True),
                log_probs.std(axis=1, keepdims=True),
                log_probs.min(axis=1, keepdims=True),
                log_probs.max(axis=1, keepdims=True),
                np.median(log_probs, axis=1, keepdims=True),],
            axis=1,
        ),
        columns=[
            "log_prob_true_thetas", 
            "top_log_prob", 
            "mean_log_prob_samples",
            "std_log_prob_samples",
            "min_log_prob_samples",
            "max_log_prob_samples",
            "median_log_prob_samples",
        ],
    )
    
    # Downcast numeric columns to save space
    df = df.apply(pd.to_numeric, downcast='integer')
    df = df.apply(pd.to_numeric, downcast='float')

    # Make sure the downcasting worked
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype(np.float16)
    
    # Save the DataFrame to a CSV file
    df.to_csv(
        output_dir / "log_probs_short_summary.csv", 
        index=False,
    )





    return df
    
        



