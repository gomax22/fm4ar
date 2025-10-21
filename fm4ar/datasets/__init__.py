
from pathlib import Path

from pydantic import BaseModel, Field
from fm4ar.datasets.dataset import load_dataset # noqa: F401

class DatasetConfig(BaseModel):
    """
    Configuration for the dataset.
    """
    name: str = Field(
        ...,
        description="Name of the dataset.",
    )

    random_seed: int = Field(
        default=42,
        description=(
            "Random seed for the data loaders: This seed controls how the "
            "dataset is split into training and validation sets. We want to "
            "be able to control this independently of, for example, the "
            "initialization of the model weights."
        ),
    )

