import pandas as pd
import numpy as np
from pathlib import Path

def save_log_probs_to_csv(
    log_probs: np.ndarray,
    top_log_probs: np.ndarray,
    log_probs_true_theta: np.ndarray,
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
            log_probs_true_theta, top_log_probs, log_probs,
            axis=1,
        ),
        columns=["log_prob_true_theta", "top_log_prob"] + [f"log_prob_sample_{i+1}" for i in range(log_probs.shape[1])],

    )
    df.to_csv(output_dir / "log_probs_summary.csv", index=False)


    return df
    
        



