"""
Unit tests for `fm4ar.time_priors`.
"""

import pytest
import torch
from fm4ar.time_priors import get_time_prior, TimePriorDistributionConfig
from fm4ar.time_priors.base import (
    TimePriorDistribution,
    PowerLawTimePrior,
    UniformTimePrior,
    LogitNormalTimePrior
)


# ==========================================
# Base Class Tests
# ==========================================
def test_time_prior_distribution_to_method():
    class DummyPrior(TimePriorDistribution):
        def sample_t(self, num_samples: int, **kwargs):
            return torch.ones(num_samples)

    prior = DummyPrior()
    assert prior.device.type == "cpu"

    new_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = prior.to(new_device)
    assert moved.device == new_device
    assert isinstance(moved, DummyPrior)


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        _ = TimePriorDistribution()  # abstract class → should fail


# ==========================================
# PowerLawTimePrior Tests
# ==========================================
def test_powerlaw_sample_shape_and_range():
    prior = PowerLawTimePrior(time_prior_exponent=1.0)
    samples = prior.sample_t(1000)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (1000,)
    assert torch.all(samples >= 0) and torch.all(samples <= 1)


def test_powerlaw_uses_kwargs_exponent_override():
    prior = PowerLawTimePrior(time_prior_exponent=1.0)
    samples_default = prior.sample_t(1000)
    samples_custom = prior.sample_t(1000, time_prior_exponent=3.0)
    # distributions should differ statistically (rough check)
    assert not torch.allclose(samples_default.mean(), samples_custom.mean())


# ==========================================
# UniformTimePrior Tests
# ==========================================
def test_uniform_sample_uniformity_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = UniformTimePrior(device=device)
    samples = prior.sample_t(500)
    assert samples.device == device
    assert (samples >= 0).all() and (samples <= 1).all()
    assert samples.shape == (500,)


# ==========================================
# LogitNormalTimePrior Tests
# ==========================================
def test_logit_normal_sample_properties():
    prior = LogitNormalTimePrior(mean=0.0, std=1.0)
    samples = prior.sample_t(1000)
    assert samples.shape == (1000,)
    assert (samples >= 0).all() and (samples <= 1).all()
    mean_val = samples.mean().item()
    assert 0.2 < mean_val < 0.8  # rough range check

def test_logit_normal_kwargs_override():
    prior = LogitNormalTimePrior(mean=0.0, std=1.0)
    s1 = prior.sample_t(1000)
    s2 = prior.sample_t(1000, mean=2.0)
    # mean shift → s2 mean should be higher
    assert s2.mean() > s1.mean()

# ==========================================
@pytest.mark.parametrize("prior_type,expected_class", [
    ("power_law", PowerLawTimePrior),
    ("uniform", UniformTimePrior),
    ("logit_normal", LogitNormalTimePrior),
])
def test_get_time_prior_returns_correct_instance(prior_type, expected_class):
    config = {
        "type": prior_type,
        "kwargs": {"time_prior_exponent": 2.0} if prior_type == "power_law" else {},
    }
    prior = get_time_prior(config)
    assert isinstance(prior, expected_class)


def test_get_time_prior_raises_for_invalid_type():
    config = {"type": "nonexistent", "kwargs": {}}
    with pytest.raises(ValueError, match="Unknown time prior distribution"):
        get_time_prior(config)


def test_time_prior_distribution_config_pydantic_validation():
    cfg = TimePriorDistributionConfig(type="uniform", kwargs={}, device="cpu")
    assert cfg.type == "uniform"
    assert isinstance(cfg.kwargs, dict)
    assert cfg.device == "cpu"
