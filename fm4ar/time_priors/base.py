"""
Define an abstract base class for time priors.
"""

import torch
from abc import ABC, abstractmethod
from typing import Any, Optional


# ==========================================
# Abstract Base Class
# ==========================================
class TimePriorDistribution(ABC):
    """
    Abstract base class for sampling time t ∈ [0, 1]
    from different prior distributions for Flow Matching.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cpu")

    @abstractmethod
    def sample_t(self, num_samples: int, **kwargs: Any) -> torch.Tensor:
        """
        Sample t from the distribution.

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Sampled values of shape (num_samples,)
        """
        pass

    def to(self, device: torch.device) -> "TimePriorDistribution":
        """
        Move the distribution sampler to a new device.
        """
        self.device = device
        return self


# ==========================================
# Power Law Prior
# ==========================================
class PowerLawTimePrior(TimePriorDistribution):
    """
    Power-law prior:
        exponent = 0 → uniform distribution
        exponent = 1 → linear bias toward t = 1
    """

    def __init__(self, time_prior_exponent: float = 1.0, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.time_prior_exponent = time_prior_exponent

    def sample_t(self, num_samples: int, **kwargs: Any) -> torch.Tensor:
        exponent = kwargs.get("time_prior_exponent", self.time_prior_exponent)
        t = torch.rand(num_samples, device=self.device)
        t = torch.pow(t, 1 / (1 + exponent))
        return t


# ==========================================
# Uniform Prior
# ==========================================
class UniformTimePrior(TimePriorDistribution):
    """
    Uniform prior over [0, 1].
    """

    def sample_t(self, num_samples: int, **kwargs: Any) -> torch.Tensor:
        return torch.rand(num_samples, device=self.device)


# ==========================================
# Logit-Normal Prior
# ==========================================
class LogitNormalTimePrior(TimePriorDistribution):
    """
    Logit-Normal prior:
        Sample x ~ N(mean, std^2)
        t = sigmoid(x)
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.mean = mean
        self.std = std

    def sample_t(self, num_samples: int, **kwargs: Any) -> torch.Tensor:
        mean = kwargs.get("mean", self.mean)
        std = kwargs.get("std", self.std)
        normal_samples = torch.normal(
            mean=mean,
            std=std,
            size=(num_samples,),
            device=self.device
        )
        return torch.sigmoid(normal_samples)


