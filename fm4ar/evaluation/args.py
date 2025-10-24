"""
Argument parser for evaluation script.
"""

import argparse  # pragma: no cover
from pathlib import Path  # pragma: no cover


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the directory containing the trained model.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "evaluate_regression_metrics",
            "evaluate_calibration_metrics",
            "evaluate_coverage_metrics",
            "draw_corner_plots"
        ],
        default=None,
        help="Stage of the evaluation workflow that should be run.",
    )
    args = parser.parse_args()

    return args
