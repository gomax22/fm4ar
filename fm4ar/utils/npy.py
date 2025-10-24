"""
Utility functions for working with HDF5 files.
"""

import torch
import numpy as np
from pathlib import Path

def save_to_npy(
    output_dir: Path,
    samples: np.ndarray,
    log_prob_samples: np.ndarray,
    log_probs_true_thetas: np.ndarray,
) -> None:
    """
    Save the given arrays to NPY files.
    Args:
        output_dir: Directory to save the NPY files to.
        samples: Samples from the posterior distribution.
        log_prob_samples: Log probabilities of the samples.
        log_probs_true_thetas: Log probabilities of the true theta values.
    """
        
    np.save(output_dir / "posterior_distribution.npy", samples)
    np.save(output_dir / "posterior_log_probs.npy", log_prob_samples)
    np.save(output_dir / "posterior_log_probs_true_thetas.npy", log_probs_true_thetas)

    # Find the sample with the highest log probability for each test sample
    log_probs = torch.from_numpy(log_prob_samples)
    posterior_samples = torch.from_numpy(samples)
    best = torch.argmax(log_probs, dim=1)
    num_test_samples = posterior_samples.shape[0]
    top_samples = posterior_samples[torch.arange(num_test_samples), best].numpy()
    top_log_probs = log_probs[torch.arange(num_test_samples), best].numpy()

    # Save the indices and the samples
    np.save(output_dir / "posterior_top_log_probs.npy", top_log_probs)
    np.save(output_dir / "posterior_top_samples_indices.npy", best)
    np.save(output_dir / "posterior_top_samples.npy", top_samples)

    return None