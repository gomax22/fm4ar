"""
Methods to prepare a new or resumed training run.
"""

from pathlib import Path

import wandb


from torch.utils.data import Dataset
from fm4ar.datasets import load_dataset
from fm4ar.models.build_model import build_model
from fm4ar.models.fmpe import FMPEModel
from fm4ar.models.npe import NPEModel
from fm4ar.torchutils.general import get_number_of_parameters
from fm4ar.training.wandb import get_wandb_id


def prepare_new(
    experiment_dir: Path,
    config: dict,
) -> tuple[FMPEModel | NPEModel, Dataset, Dataset, Dataset]:
    """
    Prepare a new training run, that is, load the dataset and initialize
    a new posterior model according to the given configuration.

    Args:
        experiment_dir: Path to the experiment directory.
        config: Full experiment configuration.

    Returns:
        A 3-tuple `(pm, train_dataset, valid_dataset)`, where `pm` is
        the posterior model, `train_dataset` is the training dataset,
        and `valid_dataset` is the validation dataset.
    """

    # Load the dataset
    print("Loading dataset...", end=" ", flush=True)
    train_dataset, valid_dataset, _ = load_dataset(config=config)
    print("Done!", flush=True)

    # Add the theta_dim and context_dim to the model settings
    config["model"]["dim_theta"] = train_dataset.dim_theta
    config["model"]["dim_context"] = train_dataset.dim_context
    config["model"]["dim_auxiliary_data"] = train_dataset.dim_auxiliary_data


    # Initialize the posterior model
    print("Building model from configuration...", end=" ", flush=True)
    model = build_model(
        experiment_dir=experiment_dir,
        config=config,
        device=config["local"]["device"],
    )
    print(f"Done! (device: {model.device})")

    # Print model summary
    print(model.network)
    n_trainable_params = get_number_of_parameters(model.network, (True,))
    n_fixed_params = get_number_of_parameters(model.network, (False,))
    n_total_params = get_number_of_parameters(model.network)
    print("Model parameters:")

    print(f"trainable: {n_trainable_params:_}")
    print(f"fixed: {n_fixed_params:_}")
    print(f"total: {n_total_params:_}")

    # Initialize Weights & Biases (if desired)
    if model.use_wandb:  # pragma: no cover
        print("\n\nInitializing Weights & Biases:", flush=True)

        # Add number of model parameters to the config
        augmented_config = config.copy()
        augmented_config["n_model_parameters"] = {
            "trainable": n_trainable_params,
            "fixed": n_fixed_params,
            "total": n_total_params,
        }

        # Add the experiment directory to the config
        augmented_config["experiment_dir"] = experiment_dir.as_posix()

        # Initialize Weights & Biases; this will produce some output to stderr
        wandb_id = get_wandb_id(experiment_dir)
        wandb.init(
            id=wandb_id,
            config=augmented_config,
            dir=experiment_dir,
            **config["local"]["wandb"],
        )

        # Define metrics
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        # Save the name of the run to a file in the experiment directory
        if wandb.run is not None:
            (experiment_dir / wandb.run.name).touch(exist_ok=True)

        print()

    return model, train_dataset, valid_dataset


def prepare_resume(
    experiment_dir: Path,
    checkpoint_name: str,
    config: dict,
) -> tuple[FMPEModel | NPEModel, Dataset, Dataset, Dataset]:
    """
    Prepare a training run by resuming from a checkpoint, that is, load
    the dataset, and instantiate the posterior model, optimizer and
    scheduler from the checkpoint.

    Args:
        experiment_dir: Path to the experiment directory.
        checkpoint_name: Name of the checkpoint file.
        config: Full experiment configuration.

    Returns:
        A tuple, `(pm, train_dataset, valid_dataset)`, where `pm` is the
        posterior model, `train_dataset` is the training dataset, and
        `valid_dataset` is the validation dataset.  
    """

    # Instantiate the posterior model
    print("Building model from checkpoint...", end=" ", flush=True)
    file_path = experiment_dir / checkpoint_name
    model = build_model(
        experiment_dir=experiment_dir,
        file_path=file_path,
        device=config["local"]["device"],
    )
    print("Done!")

    # Load the dataset (using config from checkpoint)
    print("Loading dataset...", end=" ", flush=True)
    train_dataset, valid_dataset, _ = load_dataset(config=model.config)
    print("Done!")

    # Print model summary
    print(model.network)
    n_trainable_params = get_number_of_parameters(model.network, (True,))
    n_fixed_params = get_number_of_parameters(model.network, (False,))
    n_total_params = get_number_of_parameters(model.network)
    print("Model parameters:")

    print(f"trainable: {n_trainable_params:_}")
    print(f"fixed: {n_fixed_params:_}")
    print(f"total: {n_total_params:_}")

    # Initialize Weights & Biases; this will produce some output to stderr
    if config["local"].get("wandb", False):  # pragma: no cover
        print("\n\nRe-initializing Weights & Biases:", flush=True)

        wandb_id = get_wandb_id(experiment_dir)
        wandb.init(
            id=wandb_id,
            resume="must",
            dir=experiment_dir,
            **config["local"]["wandb"],
        )

    return model, train_dataset, valid_dataset
