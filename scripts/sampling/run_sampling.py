"""
Run sampling using a trained ML model: either a "proper"
posterior model (FMPE or NPE)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from socket import gethostname

import torch
import numpy as np

from fm4ar.sampling.config import (
    SamplingConfig,
    load_config,
)
from fm4ar.sampling.args import get_cli_arguments
from fm4ar.sampling.proposals import draw_samples

from fm4ar.torchutils.general import get_cuda_info
from fm4ar.utils.hdf import load_from_hdf, save_to_hdf
from fm4ar.utils.htcondor import (
    DAGManFile,
    HTCondorConfig,
    check_if_on_login_node,
    condor_submit_dag,
    create_submission_file,
)
from fm4ar.utils.npy import save_to_npy
from fm4ar.utils.paths import expand_env_variables_in_path

def prepare_and_launch_dag(
    args: argparse.Namespace,
    config: SamplingConfig,
) -> None:
    """
    Prepare and launch the DAGMan file for running the importance
    sampling workflow on HTCondor.

    Args:
        args: The command line arguments.
        config: The importance sampling configuration.
    """

    # Initialize a new DAGMan file
    dag = DAGManFile()

    # Add jobs for the different stages of the importance sampling workflow
    for i, (stage, depends_on) in enumerate(
        [
            ("draw_samples", None),
            ("export_samples", ["draw_samples"]),
        ],
        start=1,
    ):

        # Collect HTCondorSettings for the stage
        htcondor_config: HTCondorConfig = getattr(config, stage).htcondor
        htcondor_config.arguments = [
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
            f"--working-dir {args.working_dir}",
            f"--stage {stage}",
        ]

        # For the stages that require parallel processing, add the job number
        # and the total number of parallel jobs as arguments; if we just take
        # this from the config file, things break down in non-parallel mode
        if stage in ("draw_samples"):
            htcondor_config.arguments += [
                "--job $(Process)",
                f"--n-jobs {htcondor_config.queue}",
            ]

        # Create submission file
        file_path = create_submission_file(
            htcondor_config=htcondor_config,
            experiment_dir=args.working_dir,
            file_name=f"{i}__{stage}.sub"
        )

        # Add the job to the DAGMan file
        dag.add_job(
            name=stage,
            file_path=file_path,
            bid=htcondor_config.bid,
            depends_on=depends_on,
        )

    # Save the DAGMan file
    file_path = args.working_dir / "0__sampling.dag"
    dag.save(file_path=file_path)

    # Submit the DAGMan file to HTCondor
    condor_submit_dag(file_path=file_path, verbose=True)


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nRUN SAMPLING\n")

    # Get the command line arguments and define shortcuts
    args = get_cli_arguments()

    # Resolve the working directory in case it is not absolute
    # We need to overwrite args because some functions expect args.working_dir
    # to be an absolute path. Not sure if there is a cleaner way to do this?
    if not args.working_dir.is_absolute():
        args.working_dir = expand_env_variables_in_path(
            Path(args.experiment_dir)
            / "sampling"
            / args.working_dir
        )

    # Make sure the working directory exists before we proceed
    if not args.working_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.working_dir}")

    # Ensure that we do not run compute-heavy jobs on the login node
    check_if_on_login_node(start_submission=args.start_submission)
    print("Running on host:", gethostname(), "\n", flush=True)

    # Load the importance sampling config
    config = load_config(experiment_dir=args.working_dir)

    # -------------------------------------------------------------------------
    # If --start-submission: Create DAG file, launch job, and exit
    # -------------------------------------------------------------------------

    if args.start_submission:
        prepare_and_launch_dag(args=args, config=config)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Stage 1: Draw samples from the proposal distribution
    # -------------------------------------------------------------------------

    if args.stage == "draw_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(1) Draw samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Check if the output file already exists
        output_file_path = (
            args.working_dir / "samples.hdf"
        )

        if output_file_path.exists():
            print("Samples exist already, skipping!\n")

        # Otherwise, we need to draw the proposal samples
        else:

            # Document the CUDA setup
            print("CUDA information:")
            for key, value in get_cuda_info().items():
                print(f"  {key + ':':<16}{value}")
            print()

            # Draw samples (this comes with its own progress bar)
            results = draw_samples(args=args, config=config)

            # Convert some arrays to float32 to save space
            for key in ("samples", "log_prob_samples"):
                results[key] = results[key].astype(np.float32)

            print("\nSaving results to HDF...", end=" ", flush=True)
            save_to_hdf(
                file_path=output_file_path,
                samples=results["samples"],
                log_prob_samples=results["log_prob_samples"],
                log_prob_theta_true=results["log_prob_theta_true"],
            )
            print("\nSaving results to NPY...", end=" ", flush=True)
            save_to_npy(
                output_dir=args.working_dir,
                samples=results["samples"],
                log_prob_samples=results["log_prob_samples"],
                log_prob_theta_true=results["log_prob_theta_true"],
            )
            print("Done!\n\n")

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\n\nThis took {time.time() - script_start:.2f} seconds.\n")
