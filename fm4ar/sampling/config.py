"""
Define a parser for importance sampling configurations, and a method
to load such a configuration from a YAML file.
"""

from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field
from yaml import safe_load

from fm4ar.utils.htcondor import HTCondorConfig


class DrawSamplesConfig(BaseModel):
    """
    Configuration for the "draw proposal samples" stage.
    """

    chunk_size: int = Field(
        default=1024,
        ge=1,
        description="Number of proposal samples to draw at once.",
    )
    n_samples: int = Field(
        ...,
        ge=1,
        description="Number of proposal samples to draw.",
    )
    use_amp: bool = Field(
        default=False,
        description="Use AMP for the proposal sampling",
    )
    
    batch_size: int = Field(
        ...,
        description="Batch size for the stage.",
    )
    shuffle: bool = Field(
        default=False,
        description="Whether to shuffle the training data at every epoch.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Whether to use pinned memory for the data loaders.",
    )

    drop_last: bool = Field(
        default=False,
        description=(
            "Whether the training dataloader should drop the last batch "
            "if it is smaller than the batch size."
        ),
    )
    n_workers: int | Literal["auto"] = Field(
        default="auto",
        description=(
            "Number of workers for the data loaders. If 'auto', the number "
            "of workers is set to n_available_cpus - 1, or 0 on a Mac."
        ),
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for the data loaders.",
    )
    htcondor: HTCondorConfig


class SamplingConfig(BaseModel):
    """
    Full configuration for a sampling run.
    """

    model_config = ConfigDict(protected_namespaces=())

    # General settings
    checkpoint_file_name: str = Field(
        default="model__best.pt",
        description="Name of the model checkpoint file to use.",
    )

    model_kwargs: dict[str, Any] = Field(
        default={},
        description=(
            "Additional keyword arguments for the posterior model. "
            "Usually, this should only be necessary for FMPE models to "
            "control the settings of the ODE solver (e.g., `tolerance`)."
        ),
    )

    loss_kwargs: dict = Field(
        default={},
        description=(
            "Additional keyword arguments for the loss function. This can "
            "be used, e.g., to control the `time_prior_exponent` per stage."
        ),
    )

    # Configuration for the individual stages
    draw_samples: DrawSamplesConfig

def load_config(
    experiment_dir: Path,
    name: str = "sampling.yaml",
) -> SamplingConfig:
    """
    Load the configuration inside the given experiment directory.
    """

    # Load the configuration file
    config_file = experiment_dir / name
    with open(config_file, "r") as file:
        config_dict = safe_load(file)

    # Construct the configuration object
    return SamplingConfig(**config_dict)
