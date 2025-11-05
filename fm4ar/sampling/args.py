"""
Argument parser for sampling script.
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
        "--job",
        type=int,
        default=0,
        help="Job number for parallel processing; must be in [0, n_jobs).",
    )
    parser.add_argument(
        "--max-timeouts",
        type=int,
        default=0,
        help=(
            "Maximum number of timeouts before the job is resubmitted on a "
            "different node. Default: 0 (= no timeouts allowed)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs. Default: 1.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "draw_samples",
            "merge_samples"
        ],
        default=None,
        help="Stage of the importance sampling workflow that should be run.",
    )
    args = parser.parse_args()

    return args
