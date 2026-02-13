"""
Load a dataset from the given experiment configuration.
"""


from torch.utils.data import Dataset

from fm4ar.datasets.scalers.theta_scalers import get_theta_scaler
from fm4ar.datasets.scalers.auxiliary_data_scalers import get_auxiliary_data_scaler
from fm4ar.datasets.scalers.flux_scalers import get_flux_scaler
from fm4ar.datasets.scalers.error_bars_scalers import get_error_bars_scaler
from fm4ar.utils.paths import expand_env_variables_in_path


def load_dataset(config: dict) -> tuple[
    Dataset, 
    Dataset, 
    Dataset
]:
    """
    Load a dataset from the given experiment configuration.

    Args:
        config (dict): The experiment configuration.
    Returns:
        tuple[Dataset, Dataset, Dataset]: The training, validation, and test datasets.
    """
    
    # Extract and very the dataset configuration
    dataset_name = config["dataset"]["name"]
    match dataset_name:
        case "inaf":
            from fm4ar.datasets.inaf import INAFDatasetConfig, load_inaf_dataset
            dataset_config = INAFDatasetConfig(**config["dataset"])
            train_dataset, valid_dataset, test_dataset = load_inaf_dataset(
                data_dir=expand_env_variables_in_path(dataset_config.data_dir),
                predict_log_temperature=dataset_config.predict_log_temperature,
                limit=dataset_config.limit,
                verbose=dataset_config.verbose,
                theta_scaler=get_theta_scaler(config.get("theta_scaler", {})),
                auxiliary_data_scaler=get_auxiliary_data_scaler(config.get("auxiliary_data_scaler", {})),
                flux_scaler=get_flux_scaler(config.get("flux_scaler", {})),
                error_bars_scaler=get_error_bars_scaler(config.get("error_bars_scaler", {})),
            )
        case "inara_subset":
            from fm4ar.datasets.inara_subset import INARASubsetConfig, load_inara_dataset
            dataset_config = INARASubsetConfig(**config["dataset"])
            train_dataset, valid_dataset, test_dataset = load_inara_dataset(
                data_dir=expand_env_variables_in_path(dataset_config.data_dir),
                targets=dataset_config.targets,
                components=dataset_config.components,
                auxiliary_data=dataset_config.auxiliary_data,
                limit=dataset_config.limit,
                verbose=dataset_config.verbose,
                theta_scaler=get_theta_scaler(config.get("theta_scaler", {})),
                auxiliary_data_scaler=get_auxiliary_data_scaler(config.get("auxiliary_data_scaler", {})),
                flux_scaler=get_flux_scaler(config.get("flux_scaler", {})),
                error_bars_scaler=get_error_bars_scaler(config.get("error_bars_scaler", {})),
            )
        
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    return train_dataset, valid_dataset, test_dataset
