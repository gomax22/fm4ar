"""
Methods for drawing samples from a proposal distribution (with optional DDP multi-GPU support).
"""

from argparse import Namespace
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast

from fm4ar.datasets import load_dataset
from fm4ar.sampling.config import SamplingConfig
from fm4ar.models.build_model import build_model, FMPEModel, NPEModel
from fm4ar.torchutils.general import (
    set_random_seed,
    check_for_nans,
    move_batch_to_device,
)
from fm4ar.torchutils.dataloaders import (
    build_dataloader,
    get_number_of_workers,
)
from fm4ar.utils.config import load_config as load_experiment_config
from fm4ar.utils.tracking import LossInfo, DistributedLossInfo


# -------------------------------------------------------------------------
# Distributed helpers
# -------------------------------------------------------------------------

def init_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize torch.distributed if not already initialized."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)
        print(f"[DDP] Process {rank} initialized (world size {world_size})")


def cleanup_distributed():
    """Gracefully clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def draw_samples(
    args: Namespace,
    config: SamplingConfig,
) -> dict[str, np.ndarray]:
    """
    Draw samples from the proposal distribution, supporting multi-GPU
    sampling via DistributedDataParallel (DDP).

    Args:
        args: Command-line arguments.
        config: Configuration for the sampling run.

    Returns:
        Dictionary with keys:
            - samples
            - log_prob_samples
            - log_probs_true_thetas
            - avg_loss
    """

    # ---------------------------------------------------------------------
    # Load experiment config and model type
    # ---------------------------------------------------------------------
    experiment_config = load_experiment_config(args.experiment_dir)
    model_type = experiment_config["model"]["model_type"]

    # ---------------------------------------------------------------------
    # Determine number of samples to draw for this job
    # ---------------------------------------------------------------------
    n_total = config.draw_samples.n_samples
    n_for_job = len(np.arange(args.job, n_total, args.n_jobs))
    print(f"Total samples: {n_total:,}")
    print(f"Samples for this job: {n_for_job:,}\n")

    # ---------------------------------------------------------------------
    # Multi-GPU (DDP) setup
    # ---------------------------------------------------------------------
    use_ddp = getattr(config.draw_samples, "use_distributed", False)
    world_size = int(torch.cuda.device_count())
    rank = int(args.local_rank) if hasattr(args, "local_rank") else 0

    if use_ddp and world_size > 1:
        print(f"Initializing distributed mode with {world_size} GPUs...")
        init_distributed(rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(experiment_config["local"].get("device", "cpu"))
        print(f"Running on device: {device}")

    # ---------------------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------------------
    print("Loading trained model...", end=" ", flush=True)
    model = build_model(
        experiment_dir=args.experiment_dir,
        file_path=args.experiment_dir / config.checkpoint_file_name,
        device=str(device),
    )
    model_kwargs = (
        {} if config.model_kwargs is None or model_type.lower() != "fmpe"
        else config.model_kwargs
    )
    model.network.eval()

    # Wrap model for DDP
    if use_ddp and world_size > 1:
        model.network = DDP(model.network.to(device), device_ids=[rank], output_device=rank)
        model.device = device
        if rank == 0:
            print("Model wrapped with DistributedDataParallel.")
    else:
        model.device = device

    # ---------------------------------------------------------------------
    # Fix global seed
    # ---------------------------------------------------------------------
    set_random_seed(config.draw_samples.random_seed + args.job)

    # ---------------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------------
    print("Loading dataset...", end=" ", flush=True)
    _, _, test_dataset = load_dataset(config=experiment_config)
    print("Done!", flush=True)

    # Build sampler (for DDP)
    sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=config.draw_samples.shuffle)
        if use_ddp and world_size > 1 else None
    )

    test_dataloader = build_dataloader(
        dataset=test_dataset,
        batch_size=config.draw_samples.batch_size,
        shuffle=(sampler is None and config.draw_samples.shuffle),
        drop_last=config.draw_samples.drop_last,
        n_workers=get_number_of_workers(config.draw_samples.n_workers),
        random_seed=config.draw_samples.random_seed + args.job,
        sampler=sampler,
    )

    # ---------------------------------------------------------------------
    # Perform sampling
    # ---------------------------------------------------------------------
    print(f"\nSampling using model ({model_type})...\n")

    results = draw_samples_from_ml_model(
        model=model,
        test_loader=test_dataloader,
        n_samples=n_for_job,
        chunk_size=config.draw_samples.chunk_size,
        model_kwargs=model_kwargs,
        loss_kwargs=config.loss_kwargs,
        use_amp=config.draw_samples.use_amp,
        distributed=(use_ddp and world_size > 1),
        rank=rank,
    )

    # Only the main process returns final results
    if use_ddp and world_size > 1:
        cleanup_distributed()
        if rank != 0:
            return {}

    return results


# -------------------------------------------------------------------------
# Core sampling logic
# -------------------------------------------------------------------------

def draw_samples_from_ml_model(
    model: FMPEModel | NPEModel,
    test_loader: DataLoader,
    n_samples: int,
    chunk_size: int = 1024,
    model_kwargs: Optional[dict[str, Any]] = None,
    loss_kwargs: Optional[dict[str, Any]] = None,
    use_amp: bool = False,
    distributed: bool = False,
    rank: int = 0,
) -> dict[str, np.ndarray]:
    """
    Draw samples from a trained ML model (NPE or FMPE), optionally distributed across multiple GPUs.
    """

    if use_amp and model.device == torch.device("cpu"):
        raise RuntimeError("Automatic mixed precision cannot be used on CPU.")

    loss_info = DistributedLossInfo(
        epoch=model.epoch,
        len_dataset=len(test_loader.dataset),
        batch_size=test_loader.batch_size,
        mode="Testing",
        print_freq=1,
        sync_times=False,  # usually unnecessary for evaluation
    )

    chunk_sizes = np.diff(np.r_[0: n_samples: chunk_size, n_samples])

    if rank == 0:
        print("Drawing samples from model posterior...\n")

    samples, log_prob_samples, log_probs_true_thetas = [], [], []

    torch.cuda.empty_cache()

    with torch.no_grad(), autocast(enabled=use_amp):
        for batch_idx, batch in enumerate(test_loader):
            loss_info.update_timer()
            theta, context, aux_data = move_batch_to_device(batch, model.device)

            # Compute test loss
            loss = model.loss(
                theta=theta,
                context=context,
                aux_data=aux_data,
                **(loss_kwargs or {}),
            )
            check_for_nans(loss, "test loss")
            loss_info.update(loss.item(), len(theta))
            loss_info.print_info(batch_idx)
            del loss

            # Log prob of true theta
            log_prob_theta_true = model.log_prob_batch(
                theta=theta.float().reshape(1, -1),
                context={k: v.repeat(1, 1) for k, v in context.items()},
                aux_data=aux_data.float().reshape(1, -1) if aux_data is not None else None,
                **(model_kwargs or {}),
            ).detach().cpu().numpy().flatten()
            log_probs_true_thetas.append(log_prob_theta_true)

            # Sampling loop
            samples_chunks, log_prob_chunks = [], []
            for n in chunk_sizes:
                chunk = model.sample_and_log_prob_batch(
                    context={k: v.repeat(n, 1) for k, v in context.items()},
                    aux_data=aux_data.repeat(n, 1) if aux_data is not None else None,
                    **(model_kwargs or {}),
                )
                samples_chunks.append(
                    test_loader.dataset.theta_scaler.inverse_tensor(chunk[0].detach().cpu()).numpy()
                )
                log_prob_chunks.append(chunk[1].detach().cpu().numpy())
                del chunk

            samples.append(np.concatenate(samples_chunks, axis=0))
            log_prob_samples.append(np.concatenate(log_prob_chunks, axis=0))

    # ---------------------------------------------------------------------
    # Synchronize loss across ranks
    # ---------------------------------------------------------------------
    if distributed:
        loss_info.synchronize_loss()

    avg_loss = loss_info.get_avg()

    if rank == 0:
        loss_info.print_final()
        print("\nSampling complete!\n")

    # Gather across ranks if distributed
    if distributed:
        samples = _gather_numpy(samples)
        log_prob_samples = _gather_numpy(log_prob_samples)
        log_probs_true_thetas = _gather_numpy(log_probs_true_thetas)

    samples = np.stack(samples, axis=0)
    log_prob_samples = np.stack(log_prob_samples, axis=0)
    log_probs_true_thetas = np.concatenate(log_probs_true_thetas, axis=0).reshape(-1, 1)

    return {
        "samples": samples,
        "log_prob_samples": log_prob_samples,
        "log_probs_true_thetas": log_probs_true_thetas,
        "avg_loss": avg_loss,
    }


# -------------------------------------------------------------------------
# Distributed helper for gathering numpy arrays
# -------------------------------------------------------------------------

def _gather_numpy(data_list):
    """
    Gather lists of numpy arrays from all ranks into one,
    preserving strict rank order for deterministic concatenation.
    """
    if not dist.is_initialized():
        return data_list

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, data_list)

    # Rank order is deterministic in all_gather_object
    if rank == 0:
        ordered = []
        for r in range(world_size):
            if gathered[r] is not None:
                ordered.extend(gathered[r])
        return ordered
    return []
