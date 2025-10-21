import pytest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

from fm4ar.sampling.proposals import (
    draw_samples,
    draw_samples_from_ml_model
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def fake_model():
    """Creates a fake ML model mimicking FMPEModel / NPEModel interface."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.epoch = 1

    # Mock network
    model.network = MagicMock()
    model.network.eval = MagicMock()

    # Mock loss: returns scalar tensor
    model.loss.return_value = torch.tensor(0.5, dtype=torch.float32)

    # Mock log_prob_batch: returns torch tensor
    model.log_prob_batch.return_value = torch.tensor([0.2])

    # Mock sample_and_log_prob_batch: returns tuple of tensors
    theta_samples = torch.randn(5, 2)
    log_probs = torch.randn(5)
    model.sample_and_log_prob_batch.return_value = (theta_samples, log_probs)

    return model


@pytest.fixture
def fake_test_loader():
    """Creates a fake DataLoader-like object with minimal attributes."""
    batch = {
        "theta": torch.randn(1, 2),
        "context": {"x": torch.randn(1, 2)},
        "aux_data": torch.randn(1, 1)
    }

    # Fake dataset and scaler
    scaler = MagicMock()
    scaler.inverse_tensor = MagicMock(side_effect=lambda t: t)
    dataset = MagicMock()
    dataset.theta_scaler = scaler

    loader = MagicMock()
    loader.__iter__.return_value = iter([batch])
    loader.dataset = dataset
    loader.batch_size = 1
    return loader


# -----------------------------------------------------------------------------
# Tests for draw_samples_from_ml_model
# -----------------------------------------------------------------------------
def test_draw_samples_from_ml_model_returns_expected_structure(fake_model, fake_test_loader):
    n_samples = 5
    chunk_size = 2

    results = draw_samples_from_ml_model(
        model=fake_model,
        test_loader=fake_test_loader,
        n_samples=n_samples,
        chunk_size=chunk_size,
        model_kwargs={},
        loss_kwargs={},
        use_amp=False
    )

    # Check the keys
    assert set(results.keys()) == {
        "samples", "log_prob_samples", "log_prob_theta_true", "avg_loss"
    }

    # Check array shapes and types
    assert isinstance(results["samples"], np.ndarray)
    assert isinstance(results["log_prob_samples"], np.ndarray)
    assert isinstance(results["log_prob_theta_true"], np.ndarray)
    assert isinstance(results["avg_loss"], float)

    # There should be at least n_samples rows in samples
    assert results["samples"].shape[0] >= n_samples


def test_draw_samples_from_ml_model_raises_with_amp_on_cpu(fake_model, fake_test_loader):
    """Ensure AMP not allowed on CPU."""
    with pytest.raises(RuntimeError):
        draw_samples_from_ml_model(
            model=fake_model,
            test_loader=fake_test_loader,
            n_samples=5,
            use_amp=True  # CPU device → should fail
        )


# -----------------------------------------------------------------------------
# Tests for draw_samples (integration-level with mocks)
# -----------------------------------------------------------------------------
@patch("fm4ar.sampling.draw_samples.build_model")
@patch("fm4ar.sampling.draw_samples.load_dataset")
@patch("fm4ar.sampling.draw_samples.build_dataloader")
@patch("fm4ar.sampling.draw_samples.get_number_of_workers", return_value=0)
@patch("fm4ar.sampling.draw_samples.set_random_seed")
@patch("fm4ar.sampling.draw_samples.draw_samples_from_ml_model")
@patch("fm4ar.sampling.draw_samples.load_experiment_config")
def test_draw_samples_end_to_end(
    mock_load_experiment_config,
    mock_draw_samples_from_ml_model,
    mock_set_seed,
    mock_get_workers,
    mock_build_dataloader,
    mock_load_dataset,
    mock_build_model,
):
    # -------------------------------------------------------------------------
    # Setup mocks
    # -------------------------------------------------------------------------
    fake_results = {
        "samples": np.ones((3, 2)),
        "log_prob_samples": np.zeros(3),
        "log_prob_theta_true": np.array([0.5]),
        "avg_loss": 0.123,
    }
    mock_draw_samples_from_ml_model.return_value = fake_results

    # Mock load_experiment_config return
    mock_load_experiment_config.return_value = {
        "model": {"model_type": "fmpe"},
        "local": {"device": "cpu"},
    }

    # Mock build_model
    fake_model = MagicMock()
    fake_model.network = MagicMock()
    fake_model.network.eval = MagicMock()
    mock_build_model.return_value = fake_model

    # Mock dataset and dataloader
    mock_load_dataset.return_value = (None, None, MagicMock())
    mock_build_dataloader.return_value = MagicMock()

    # -------------------------------------------------------------------------
    # Build args and config
    # -------------------------------------------------------------------------
    args = SimpleNamespace(
        experiment_dir=Path("fake_experiment"),
        job=0,
        n_jobs=1
    )

    draw_cfg = SimpleNamespace(
        n_samples=5,
        random_seed=123,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        n_workers=0,
        chunk_size=2,
        use_amp=False
    )

    config = SimpleNamespace(
        draw_samples=draw_cfg,
        checkpoint_file_name="checkpoint.pt",
        model_kwargs=None,
        loss_kwargs={}
    )

    # -------------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------------
    results = draw_samples(args, config)

    # -------------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------------
    mock_build_model.assert_called_once()
    mock_load_dataset.assert_called_once()
    mock_build_dataloader.assert_called_once()
    mock_draw_samples_from_ml_model.assert_called_once()

    # Ensure return structure matches mocked data
    assert results == fake_results
