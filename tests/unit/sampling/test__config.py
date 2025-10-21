"""
Unit tests for `fm4ar.sampling.config`.
"""

from fm4ar.sampling.config import load_config
from fm4ar.utils.paths import get_experiments_dir


def test__load_config() -> None:
    """
    Test `load_config`.
    """

    # Load the template config
    experiment_dir = (
        get_experiments_dir() / "templates" / "sampling"
    )
    config = load_config(experiment_dir)

    # Check that the config is loaded correctly
    assert config.checkpoint_file_name == "model__best.pt"
    assert config.draw_samples.chunk_size == 1024
