"""
Convenience function for loading priors from a config.
"""

from fm4ar.time_priors.base import TimePriorDistribution, LogitNormalTimePrior, PowerLawTimePrior, UniformTimePrior
from pydantic import BaseModel, Field


class TimePriorDistributionConfig(BaseModel):
    """
    Configuration for a data transform.
    """

    type: str = Field(
        ...,
        description="Type of the time prior distribution."
        )
    kwargs: dict = Field(
        {},
        description="Keyword arguments for the time prior distribution.",
    )

    device: str = Field(
        ...,
        description="Device to use for the time prior distribution.",
    )



def get_time_prior(
    config: TimePriorDistributionConfig # experiment config
) -> TimePriorDistribution:
    """
    Return a time prior distribution instance based on its name.
    Args:
        config: Configuration for the time prior distribution.
    Returns:
        An instance of the requested time prior distribution.
    """
    time_prior_type = config.get('type', 'uniform')
    time_prior_kwargs = config.get('kwargs', {})

    match time_prior_type:
        case "power_law":
            return PowerLawTimePrior(**time_prior_kwargs)
        case "uniform":
            return UniformTimePrior(**time_prior_kwargs)
        case "logit_normal":
            return LogitNormalTimePrior(**time_prior_kwargs)
        case _:
            raise ValueError(f"Unknown time prior distribution: {time_prior_type}")