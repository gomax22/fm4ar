"""
Methods for drawing samples from a proposal distribution.
"""

from argparse import Namespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


from fm4ar.datasets import load_dataset
from fm4ar.sampling.config import SamplingConfig
from fm4ar.models.build_model import build_model, FMPEModel, NPEModel
from fm4ar.torchutils.general import (
    set_random_seed, 
    check_for_nans, 
    move_batch_to_device
)
from fm4ar.torchutils.dataloaders import (
    build_dataloader,
    get_number_of_workers,
)
from fm4ar.utils.config import load_config as load_experiment_config
from fm4ar.utils.tracking import LossInfo

def draw_samples(
    args: Namespace,
    config: SamplingConfig,
) -> dict[str, np.ndarray]:
    """
    Draw samples from the proposal distribution.

    Args:
        args: Command-line arguments.
        config: Configuration for the sampling run.

    Returns:
        A dictionary containing:
        (1) the log probability of the true theta under the proposal
            distribution,
        (2) the samples drawn from the proposal distribution, and
        (3) the log probability of the samples under the proposal
            distribution.
        (4) the average loss on the test dataset.
    """

    # Determine the model type: FMPE / NPE or unconditional flow?
    experiment_config = load_experiment_config(args.experiment_dir)
    model_type = experiment_config["model"]["model_type"]

    # Determine the number of samples that the current job should draw
    n_total = config.draw_samples.n_samples
    n_for_job = len(np.arange(args.job, n_total, args.n_jobs))
    print(f"Total number of samples to draw:             {n_total:,}")
    print(f"Number of samples to draw for current job:   {n_for_job:,}")
    print()


    # TODO: load scalers, load dataset, load model, draw samples
    # Load the trained model
    print("Loading trained model...", end=" ")
    model = build_model(
        experiment_dir=args.experiment_dir,
        file_path=args.experiment_dir / config.checkpoint_file_name,
        device=experiment_config["local"].get("device", "cpu"),
    )
    model_kwargs = (
        {} if config.model_kwargs is None or model_type != "fmpe" 
        else config.model_kwargs
    )
    model.network.eval()
    print("Done!")

    # Fix the global seed for PyTorch.
    # Note: This needs to happen *after* loading the model, because the model
    # constructor will set the random seed itself (see `initialize_network()`).
    # Clearly, this is all not very ideal, but it seems that at least the
    # libraries used for the NPE models do not provide a way to set the seed
    # for the RNG for the model itself but rely on the global state...
    set_random_seed(config.draw_samples.random_seed + args.job)

    # Load the dataset (to get the context for the model)
    print("Loading dataset...", end=" ", flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    print("Done!", flush=True)

    # Build dataloader for the test dataset
    test_dataloader = build_dataloader(
        dataset=test_dataset,
        batch_size=config.draw_samples.batch_size,
        shuffle=config.draw_samples.shuffle,
        drop_last=config.draw_samples.drop_last,
        n_workers=get_number_of_workers(config.draw_samples.n_workers),
        random_seed=config.draw_samples.random_seed + args.job,
    )

    # Draw samples either from an FMPE / NPE model...
    print(f"Running for ML model ({model_type})!\n")
    results = draw_samples_from_ml_model(
        model=model,
        test_loader=test_dataloader,
        n_samples=n_for_job,
        chunk_size=config.draw_samples.chunk_size,
        model_kwargs=model_kwargs,
        loss_kwargs=config.loss_kwargs,
        use_amp=config.draw_samples.use_amp,
    )

    return results


def draw_samples_from_ml_model(
    model: FMPEModel | NPEModel,
    test_loader: DataLoader,
    n_samples: int,
    chunk_size: int = 1024,
    model_kwargs: dict[str, Any] | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    use_amp: bool = False,
) -> dict[str, np.ndarray]:
    """
    Draw samples from a trained ML model (NPE or FMPE),
    and also evaluate the log-probability of the true theta.

    Args:
        model: The trained ML model (NPE or FMPE).
        test_loader: DataLoader for the test dataset.
        n_samples: Number of samples to draw from the model.
        chunk_size: Size of the "chunks" for drawing samples (i.e., the
            number of samples drawn at once). This is used to avoid
            running out of GPU memory.
        model_kwargs: Additional keyword arguments for the model.
            This is useful for specifying the tolerance for FMPE models.
        loss_kwargs: Additional keyword arguments for the loss function.
        use_amp: If True, use automatic mixed-precision for the model.
            This only makes sense for FMPE models.
    """

    # Check if we can use automatic mixed precision
    if use_amp and model.device == torch.device("cpu"):
        raise RuntimeError(  # pragma: no cover
            "Don't use automatic mixed precision on CPU!"
        )

    # Set up a LossInfo object to keep track of the loss and times
    loss_info = LossInfo(
        epoch=model.epoch,
        len_dataset=len(test_loader.dataset),  # type: ignore
        batch_size=int(test_loader.batch_size),  # type: ignore
        mode="Testing",
        print_freq=1,
    )

    example_batch = next(iter(test_loader))
    theta_example, _, _ = move_batch_to_device(example_batch, model.device)
    n_test_batches = len(test_loader)
    batch_size = theta_example.shape[0]
    theta_dim = theta_example.shape[1]

    samples = torch.zeros((len(test_loader.dataset), n_samples, theta_dim), dtype=torch.float32)
    log_prob_samples = torch.zeros((len(test_loader.dataset), n_samples), dtype=torch.float32)
    log_probs_true_thetas = torch.zeros((len(test_loader.dataset), 1), dtype=torch.float32)

    # Determine the chunk sizes: Every chunk should have `chunk_size` samples,
    # except for the last one, which may have fewer samples.
    chunk_sizes = np.diff(np.r_[0: n_samples: chunk_size, n_samples])

    # Draw samples from the model posterior ("proposal distribution")
    print("Drawing samples from the model posterior:", flush=True)

    # -------------------------------------------------------------------------
    # Compute test loss
    # -------------------------------------------------------------------------

    # Clear the cache to reduce out-of-memory errors
    torch.cuda.empty_cache()

    with torch.no_grad() and autocast(enabled=use_amp):
        start_idx = 0

        # Iterate over the batches
        batch: dict[str, torch.Tensor]
        for batch_idx, batch in enumerate(test_loader):
            
            loss_info.update_timer()

            # Move data to device
            theta, context, aux_data = move_batch_to_device(batch, model.device)
            bsz = theta.shape[0]
            
            # Compute test loss
            loss = model.loss(
                theta=theta,
                context=context,
                aux_data=aux_data,
                **loss_kwargs,
            )
            check_for_nans(loss, "test loss")

            # Update loss for history and logging
            loss_info.update(loss.item(), len(theta))
            loss_info.print_info(batch_idx)

            # Explicitly delete the loss; otherwise we get a major memory
            # leak on the GPU (at least for the NPE model)
            del loss

            # -------------------------------------------------------------------------
            # Compute log prob of true theta under the model
            # -------------------------------------------------------------------------
            log_prob_theta_true = model.log_prob_batch(
                theta=theta.float(),
                context=context,
                aux_data=aux_data.float() if aux_data is not None else None,
                **model_kwargs,
            ).detach().cpu()


            log_probs_true_thetas[start_idx:start_idx + bsz, 0] = log_prob_theta_true

            # Initialize lists to store the samples and log-probs for this batch
            all_samples = []
            all_log_probs = []

            for n in chunk_sizes:

                # Adjust the size of both the context and auxiliary data so that the batch size matches
                # the desired chunk size, and move it to the correct device
                # Draw samples and corresponding log-probs from the model
                chunk = model.sample_and_log_prob_batch(
                    context={
                        k: v.repeat_interleave(n, dim=0)
                        for k, v in context.items()
                    },
                    aux_data=(
                        aux_data.repeat_interleave(n, dim=0) 
                        if aux_data is not None else None
                    ),
                    **model_kwargs,
                )

                # Reshape back to [batch_size, n, theta_dim]
                all_samples.append(
                    test_loader
                    .dataset
                    .theta_scaler
                    .inverse_tensor(chunk[0].detach().cpu())
                    .view(bsz, n, theta_dim)
                )
                all_log_probs.append(
                    chunk[1]
                    .detach()
                    .cpu()
                    .view(bsz, n)
                )

                del chunk

            # Concatenate across chunks -> [batch_size, n_samples, ...]
            batch_samples = torch.cat(all_samples, dim=1)
            batch_log_probs = torch.cat(all_log_probs, dim=1)

            # Store into preallocated arrays
            samples[start_idx:start_idx + bsz] = batch_samples
            log_prob_samples[start_idx:start_idx + bsz] = batch_log_probs

            start_idx += bsz

    print(flush=True)
    print("Done!\n")

    return {
        "samples": samples.numpy(),
        "log_prob_samples": log_prob_samples.numpy(),
        "log_probs_true_thetas": log_probs_true_thetas.numpy(),
        "avg_loss": loss_info.get_avg(),
    }


