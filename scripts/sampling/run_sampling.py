"""
Run sampling using a trained ML model: either a "proper"
posterior model (FMPE or NPE)
"""

import time
import torch
import torch.multiprocessing as mp
import numpy as np

from math import ceil
from fm4ar.sampling.config import (
    load_config,
)
from fm4ar.sampling.args import get_cli_arguments
from fm4ar.sampling.proposals import draw_samples

from fm4ar.torchutils.general import get_cuda_info
from fm4ar.utils.hdf import (
    load_from_hdf, 
    save_to_hdf, 
    merge_hdf_files, 
)
from fm4ar.utils.nfe import merge_histories, NFEProfiler
from fm4ar.utils.npy import save_to_npy
from pprint import pprint

# -------------------------------------------------------------------------
# Define worker function at module level
# -------------------------------------------------------------------------
def run_on_gpu(rank, args, config, job_ids):
    """
    Run one sampling job on a given GPU.
    Each spawn round gets its own subset of jobs.
    """
    n_gpus = torch.cuda.device_count()
    gpu_id = rank % n_gpus
    torch.cuda.set_device(gpu_id)

    job_id = job_ids[rank]
    print(f"[GPU {gpu_id}] Starting sampling job {job_id}")

    args.job = job_id

    output_file_path = args.experiment_dir / f"samples-{job_id:04d}.hdf"
    if output_file_path.exists():
        print(f"[GPU {gpu_id}] {output_file_path.name} exists already, skipping.\n")
        return

    results = draw_samples(args=args, config=config)
    profiler = results.pop("profiler")

    # Convert some arrays to float32 to save space
    for key in ("samples", "log_prob_samples", "log_probs_true_thetas"):
        results[key] = results[key].astype(np.float32)

    print("\nSaving results to HDF...", end=" ", flush=True)
    save_to_hdf(
        file_path=output_file_path,
        samples=results["samples"],
        log_prob_samples=results["log_prob_samples"],
        log_probs_true_thetas=results["log_probs_true_thetas"],
    )
    print("Done.\n")


    print("Showing NFE profiler summary:")
    pprint(profiler.summary())
    print("Done.\n")

    print("Exporting NFE profiler data...", end=" ", flush=True)
    # TODO: make output format configurable
    output_file_path = args.experiment_dir / f"nfe-profile-{job_id:04d}.pkl"
    profiler.export(
        file_path=output_file_path,
    )
    print("Done.\n")
    print(f"[GPU {gpu_id}] Finished job {job_id}.\n")
    return

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nRUN SAMPLING\n")

    # Get the command line arguments and define shortcuts
    args = get_cli_arguments()

    # Make sure the working directory exists before we proceed
    if not args.experiment_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.experiment_dir}")

    # Load the sampling config
    config = load_config(experiment_dir=args.experiment_dir)

    # -------------------------------------------------------------------------
    # Stage 1: Draw samples from the proposal distribution
    # -------------------------------------------------------------------------
    if args.stage == "draw_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(1) Draw samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Detect GPUs
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available!")

        print(f"Detected {n_gpus} GPUs.")
        # Document the CUDA setup
        print("CUDA information:")
        for key, value in get_cuda_info().items():
            print(f"  {key + ':':<16}{value}")
        print()

        # Define total jobs
        n_jobs = args.n_jobs
        all_job_ids = list(range(n_jobs))
        n_rounds = ceil(n_jobs / n_gpus)
        print(f"Total jobs: {n_jobs} | Running in {n_rounds} rounds of {n_gpus} GPUs each.\n")

        # Run sampling in rounds
        for round_idx in range(n_rounds):
            start = round_idx * n_gpus
            end = min(start + n_gpus, n_jobs)
            job_ids = all_job_ids[start:end]

            print(f"\n=== Round {round_idx + 1}/{n_rounds}: Running jobs {job_ids} ===\n")
            mp.spawn(run_on_gpu, args=(args, config, job_ids), nprocs=len(job_ids), join=True)


    # -------------------------------------------------------------------------
    # Stage 2: Merge samples from the proposal distribution
    # -------------------------------------------------------------------------

    if args.stage == "merge_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(2) Merge samples from posterior distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        print("Merging HDF files:", flush=True)
        merge_hdf_files(
            target_dir=args.experiment_dir,
            name_pattern="samples-*.hdf",
            output_file_path=args.experiment_dir / "samples.hdf",
            keys=["samples", "log_prob_samples"],
            singleton_keys=["log_probs_true_thetas"],
            delete_after_merge=config.merge_samples.delete_after_merge,
            show_progressbar=config.merge_samples.show_progressbar,
            axis=1,
        )
        print("\n")

        print("Loading merged HDF file...", flush=True)
        data = load_from_hdf(
            file_path=args.experiment_dir / "samples.hdf",
            keys=["samples", "log_prob_samples", "log_probs_true_thetas"],
        )
        print("Done.\n")

        print("Converting merged HDF to NPY files...", flush=True)
        save_to_npy(
            output_dir=args.experiment_dir,
            samples=data["samples"],
            log_prob_samples=data["log_prob_samples"],
            log_probs_true_thetas=data["log_probs_true_thetas"],
        )
        print("Done.\n")

    # -------------------------------------------------------------------------
    # Stage 3: Merge profiler data
    # -------------------------------------------------------------------------

    if args.stage == "merge_profilers" or args.stage is None:

        print(80 * "-", flush=True)
        print("(3) Merge NFE Profiler data", flush=True)
        print(80 * "-" + "\n", flush=True)

        print("Merging NFE profiler data...", flush=True)
        # TODO: make output format configurable
        all_histories = merge_histories(
            target_dir=args.experiment_dir,
            name_pattern="nfe-profile-*.pkl",
            output_file_path=args.experiment_dir / "nfe-profile.pkl",
            reindex_batches=config.merge_profilers.reindex_batches,
            delete_after_merge=config.merge_profilers.delete_after_merge,
            show_progressbar=config.merge_profilers.show_progressbar,

        )
        print("Done.\n")

        print("Creating merged NFE profiler from history...", flush=True)
        profiler = NFEProfiler.from_history(all_histories)
        print("Done.\n")

        # Print summary
        if profiler is not None:
            print("Showing merged NFE profiler summary:")
            pprint(profiler.summary())
            print("Done.\n")
        else:
            print("No profiler data found to merge.\n")    
    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\n\nThis took {time.time() - script_start:.2f} seconds.\n")
