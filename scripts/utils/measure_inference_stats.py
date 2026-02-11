"""
Measure the inference time of the model.
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from yaml import safe_load

from fm4ar.sampling.config import (
    SamplingConfig,
    load_config,
)

from fm4ar.datasets import load_dataset
from fm4ar.models.build_model import build_model, FMPEModel
from fm4ar.torchutils.general import (
    set_random_seed, 
    move_batch_to_device
)
from fm4ar.torchutils.dataloaders import (
    build_dataloader,
    get_number_of_workers,
)
from fm4ar.utils.config import load_config as load_experiment_config

def get_checkpoint_file_name(
    experiment_dir: Path,
) -> str | None:
    checkpoint_file_name = (
        experiment_dir / "model__best.pt" 
        if (experiment_dir / "model__best.pt").exists() 
        else None
    )
    checkpoint_file_name = (
        experiment_dir / "model__latest.pt"
        if checkpoint_file_name is None and (experiment_dir / "model__latest.pt").exists() 
        else checkpoint_file_name
    ) 
    checkpoint_file_name = (
        experiment_dir / "model__initial.pt"
        if checkpoint_file_name is None and (experiment_dir / "model__initial.pt").exists() 
        else checkpoint_file_name
    )

    return checkpoint_file_name
    


if __name__ == "__main__":

    script_start = time()
    print("\nMEASURE INFERENCE TIME\n")

    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help=(
            "Path to the configuration file specifying the setup for the "
            "timing experiment (model, target spectrum, n_samples, ...)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit the number of samples to use for the timing experiment. "
            "This can be useful for debugging or if the timing experiment takes too long."
        ),
    )
    args = parser.parse_args()
    
    # Make sure the working directory exists before we proceed
    if not args.experiment_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.experiment_dir}")
    
    # Get the device (running this script without a GPU does not make sense)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    

    # Load the sampling config
    config: SamplingConfig = load_config(experiment_dir=args.experiment_dir)

    # Determine the model type: FMPE / NPE or unconditional flow?
    experiment_config = load_experiment_config(args.experiment_dir)
    experiment_config["dataset"]["limit"] = args.limit
    model_type = experiment_config["model"]["model_type"]

    # Load the dataset (to get the context for the model)
    print("Loading dataset...", end=" ", flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    print("Done!", flush=True)

    # Add the theta_dim and context_dim to the model settings
    experiment_config["model"]["dim_theta"] = test_dataset.dim_theta
    experiment_config["model"]["dim_context"] = test_dataset.dim_context
    experiment_config["model"]["dim_auxiliary_data"] = test_dataset.dim_auxiliary_data

    # Build dataloader for the test dataset
    test_dataloader = build_dataloader(
        dataset=test_dataset,
        batch_size=1, 
        # batch_size=config.draw_samples.batch_size,
        shuffle=config.draw_samples.shuffle,
        drop_last=config.draw_samples.drop_last,
        n_workers=get_number_of_workers(config.draw_samples.n_workers),
        random_seed=config.draw_samples.random_seed,
    )

    # Load the trained model
    # Measuring inference times should be a good practice to do before starting a training run, 
    # This makes us sure that 
    # ( i) the model can run on the target hardware and,
    # (ii) we can get an estimate of how long sampling will take.
    # This is especially important for the FMPE model and particularly for adaptive ODE solvers.
    # Clearly, before training the model, we cannot use the checkpoint file to make a precise estimate.
    # However, we can still load the model architecture and measure the inference time with random weights, 
    # hopefully getting a rough estimate of the inference time.
    print("Loading trained model...", end=" ")
    model = build_model(
        experiment_dir=args.experiment_dir,
        file_path=get_checkpoint_file_name(args.experiment_dir),
        config=experiment_config,
        device=experiment_config["local"].get("device", "cpu"),
    )
    model_kwargs = (
        {} if config.model_kwargs is None or model_type != "fmpe" 
        else config.model_kwargs
    )
    model.network.eval()
    print("Done!")

    set_random_seed(config.draw_samples.random_seed)

    # Sanity check: number of samples should be a multiple of the chunk size
    # so that we can use the same context dictionary for all chunks
    n_samples = config.draw_samples.n_samples
    chunk_size = config.draw_samples.chunk_size
    if n_samples % chunk_size != 0:
        raise ValueError("n_samples must be a multiple of chunksize!")

    # Determine the chunk sizes: Every chunk should have `chunk_size` samples,
    # except for the last one, which may have fewer samples.
    chunk_sizes = np.diff(np.r_[0: n_samples: chunk_size, n_samples])

    # Clear the cache to reduce out-of-memory errors
    torch.cuda.empty_cache()

    # Use automatic mixed precision for the FMPE model
    use_amp = isinstance(model, FMPEModel)

    # Benchmark the `sample()` and `sample_and_logprob()` method
    # We always skip the first run to avoid any initialization overhead
    times: dict[str, list[float]] = {}
    vram_usage: dict[str, list[float]] = {}
    for label, method, kwargs in [
        (
            "sample_and_log_prob",
            model.sample_and_log_prob_batch,
            model_kwargs # config["model"]["sample_and_logprob_kwargs"],
        ),
        (
            "sample",
            model.sample_batch,
            model_kwargs # config["model"]["sample_kwargs"],
        ),
    ]:
        # TODO: make an estimate of VRAM usage and print it here, to make sure it fits on the target GPU
        print(f"Benchmarking `{label}()`:\n", flush=True)
        times[label] = []
        vram_usage[label] = []
        sample_counter = 0
        with torch.no_grad() and autocast(enabled=use_amp):
            
            batch: dict[str, torch.Tensor]
            for i, batch in enumerate(test_dataloader):
                # Move data to device
                theta, context, aux_data = move_batch_to_device(batch, model.device)
                batch_size = theta.shape[0]

                for j, n in enumerate(chunk_sizes):
                    start_time = time()
                    
                    # Repeat the context to match the desired chunk size and move it to the GPU
                    chunk_context = {
                        k: v.repeat_interleave(n, dim=0)
                        for k, v in context.items()
                    }
                    chunk_aux_data = (
                        aux_data.repeat_interleave(n, dim=0) 
                        if aux_data is not None else None
                    )

                    method(context=chunk_context, aux_data=chunk_aux_data, **kwargs)
                    # print(f"[{chunk_sizes[:j+1].sum():3d}/{n_samples:3d}] Done!", flush=True)

                total_time = time() - start_time
                vram = torch.cuda.memory_allocated() / 1024**3

                times[label].append(total_time)
                vram_usage[label].append(vram)

                sample_counter += batch_size
                print(f"[{sample_counter}/{len(test_dataset)}] Total time: {total_time:.3f} s, VRAM usage: {vram} GB\n", flush=True)
            print(
                (
                    f"\nMean Inference Time: {np.mean(times[label]):.3f}\n"
                    f"Median Inference Time: {np.median(times[label]):.3f}\n"
                    f"Std. Inference Time:   {np.std(times[label]):.3f}\n"
                ),
                flush=True,
            )
            print(
                (
                    f"\nMean VRAM usage: {np.mean(vram_usage[label]):.3f} GB\n"
                    f"Median VRAM usage: {np.median(vram_usage[label]):.3f} GB\n"
                    f"Std. VRAM usage:   {np.std(vram_usage[label]):.3f} GB\n"
                ),
                flush=True,
            )

    # Construct a single DataFrame with the results
    # Report also models kwargs (e.g. integration method, num_steps or tolerance)
    df = pd.DataFrame({
        "method": [label for label in times.keys() for _ in times[label]],
        "inference_time": [t for label in times.keys() for t in times[label]],
        "vram_usage": [v for label in vram_usage.keys() for v in vram_usage[label]],
        "method": [model_kwargs.get("method", "N/A") for label in times.keys() for _ in times[label]],
        "num_steps": [model_kwargs.get("num_steps", "N/A") for label in times.keys() for _ in times[label]],
        "tolerance": [model_kwargs.get("tolerance", "N/A") for label in times.keys() for _ in times[label]],
    })
    df.to_csv(
        args.experiment_dir / "estimated_inference_time_and_vram_usage.csv", 
        index=False
    )
    


    # df = pd.DataFrame(times)
    # df.to_csv(
    #     args.experiment_dir / "estimated_inference_time.csv", 
    #     index=False
    # )

    # df = pd.DataFrame(vram_usage)
    # df.to_csv(
    #     args.experiment_dir / "estimated_vram_usage.csv", 
    #     index=False
    # )
    # Print the total runtime
    print(f"This took {time() - script_start:.1f} seconds!\n")
