"""
Define a parser for evaluation, and a method
to load such a configuration from a YAML file.
"""

from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field
from yaml import safe_load

class CalibrationConfig(BaseModel):
    """
    Configuration for the "evaluate_calibration_metrics" stage.
    """
    bins: int = Field(
        default=20,
        description="Number of bins for calibration histograms.",
    )

    start_quantiles: float = Field(
        default=0.05,
        description="Start quantile for calibration diagrams.",
    )
    end_quantiles: float = Field(
        default=0.95,
        description="End quantile for calibration diagrams.",
    )
    num_quantiles: int = Field(
        default=19,
        description="Number of quantiles for calibration diagrams.",
    )
    
    

class RegressionConfig(BaseModel):
    """
    Configuration for the "evaluate_regression_metrics" stage.
    """

    figsize: tuple[int, int] = Field(
        default=(8, 6),
        description="Figure size for plots.",
    )

    fontsize: int = Field(
        default=14,
        description="Font size for plots.",
    )

    y_scale: str = Field(
        default='log',
        description="Y-axis scale for plots.",
    )



class CoverageConfig(BaseModel):
    """
    Configuration for the "evaluate_coverage_metrics" stage.
    """
    confidence_levels: List[float] = Field(
        default=[0.68, 0.95, 0.99],
        description="List of confidence levels for coverage evaluation.",
    )
    support: bool = Field(
        default=True,
        description="Whether to compute support coverage.",
    )

    statistics: List[str] = Field(
        default=['posterior_mean', 'posterior_std', 'posterior_accuracy'],
        description="List of statistics for coverage evaluation.",
    )


class LogProbsConfig(BaseModel):
    """
    Configuration for the "evaluate_log_probs" stage.
    """
    key: str = Field(
        ...,
        description="Placeholder key for log probability evaluation.",
    )

class DrawCornerPlotsConfig(BaseModel):
    """
    Configuration for the "draw_corner_plots" stage.
    """

    max_plots: int = Field(
        default=10,
        description="Maximum number of corner plots to draw.",
    )

    label_kwargs: dict[str, Any] = Field(
        default={
            "fontsize": 12,
            "fontname": "Times New Roman",
        },
        description="Keyword arguments for label styling in corner plots.",
    )

    title_kwargs: dict[str, Any] = Field(
        default={
            "fontsize": 10,
            "fontname": "Times New Roman",
        },
        description="Keyword arguments for title styling in corner plots.",
    )

    legend_kwargs: dict[str, Any] = Field(
        default={
            "fontsize": 20,
            "fontname": "Times New Roman",
        },
        description="Keyword arguments for legend styling in corner plots.",
    )
    offset: float = Field(
        default=0.2,
        description="Offset for axis limits around true values in corner plots.",
    )


class EvaluationConfig(BaseModel):
    """
    Full configuration for an evaluation run.
    """

    # Configuration for the individual stages
    evaluate_calibration_metrics: CalibrationConfig
    evaluate_regression_metrics: RegressionConfig
    evaluate_coverage_metrics: CoverageConfig
    evaluate_log_probs: LogProbsConfig
    draw_corner_plots: DrawCornerPlotsConfig

def load_config(
    experiment_dir: Path,
    name: str = "evaluation.yaml",
) -> EvaluationConfig:
    """
    Load the configuration inside the given experiment directory.
    """

    # Load the configuration file
    config_file = experiment_dir / name
    with open(config_file, "r") as file:
        config_dict = safe_load(file)

    # Construct the configuration object
    return EvaluationConfig(**config_dict)
