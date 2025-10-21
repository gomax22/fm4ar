import pytest
import numpy as np
import torch
from pathlib import Path
from fm4ar.utils.npy import save_to_npy


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Fixture to create a temporary directory for file output."""
    return tmp_path


def test_save_to_npy_creates_expected_files(tmp_output_dir):
    # ----------------------------------------
    # Create synthetic test data
    # ----------------------------------------
    num_test_samples = 3
    num_posterior_samples = 4
    dim = 2

    # samples shape: (num_test_samples, num_posterior_samples, dim)
    samples = np.random.randn(num_test_samples, num_posterior_samples, dim).astype(np.float32)
    log_prob_samples = np.random.randn(num_test_samples, num_posterior_samples).astype(np.float32)
    log_prob_theta_true = np.random.randn(num_test_samples).astype(np.float32)

    # ----------------------------------------
    # Call function
    # ----------------------------------------
    save_to_npy(
        output_dir=tmp_output_dir,
        samples=samples,
        log_prob_samples=log_prob_samples,
        log_prob_theta_true=log_prob_theta_true,
    )

    # ----------------------------------------
    # Check file creation
    # ----------------------------------------
    expected_files = [
        "posterior_distribution.npy",
        "posterior_log_probs.npy",
        "posterior_log_probs_true_theta.npy",
        "posterior_top_samples_indices.npy",
        "posterior_top_samples.npy",
    ]

    for fname in expected_files:
        fpath = tmp_output_dir / fname
        assert fpath.exists(), f"{fname} was not created."

    # ----------------------------------------
    # Check saved array contents
    # ----------------------------------------
    loaded_samples = np.load(tmp_output_dir / "posterior_distribution.npy")
    loaded_log_probs = np.load(tmp_output_dir / "posterior_log_probs.npy")
    loaded_log_probs_true_theta = np.load(tmp_output_dir / "posterior_log_probs_true_theta.npy")
    loaded_best_indices = np.load(tmp_output_dir / "posterior_top_samples_indices.npy")
    loaded_top_samples = np.load(tmp_output_dir / "posterior_top_samples.npy")

    # Match shapes and values
    np.testing.assert_allclose(loaded_samples, samples)
    np.testing.assert_allclose(loaded_log_probs, log_prob_samples)
    np.testing.assert_allclose(loaded_log_probs_true_theta, log_prob_theta_true)

    # ----------------------------------------
    # Validate the "best sample" logic
    # ----------------------------------------
    torch_best = torch.argmax(torch.from_numpy(log_prob_samples), dim=1)
    expected_top_samples = samples[np.arange(num_test_samples), torch_best.numpy()]

    np.testing.assert_array_equal(loaded_best_indices, torch_best.numpy())
    np.testing.assert_allclose(loaded_top_samples, expected_top_samples)


def test_save_to_npy_handles_single_sample(tmp_output_dir):
    """Edge case: when there's only one posterior sample per test case."""
    samples = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)  # shape (2,1,2)
    log_prob_samples = np.array([[0.5], [0.1]], dtype=np.float32)
    log_prob_theta_true = np.array([0.2, 0.3], dtype=np.float32)

    save_to_npy(tmp_output_dir, samples, log_prob_samples, log_prob_theta_true)

    # Even in this edge case, "best index" should always be zero
    best_indices = np.load(tmp_output_dir / "posterior_top_samples_indices.npy")
    top_samples = np.load(tmp_output_dir / "posterior_top_samples.npy")

    np.testing.assert_array_equal(best_indices, np.zeros(2, dtype=np.int64))
    np.testing.assert_allclose(top_samples, samples[:, 0])
