"""
Load a dataset from the given experiment configuration.
"""


from torch.utils.data import Dataset

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.datasets.auxiliary_data_scalers import get_auxiliary_data_scaler
from fm4ar.utils.paths import expand_env_variables_in_path


def load_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load a dataset from the given experiment configuration.
    """
    
    # Extract and very the dataset configuration
    dataset_name = config["dataset"]["name"]
    match dataset_name:
        case "vasist_2023":
            from fm4ar.datasets.vasist_2023 import VasistDatasetConfig, load_vasist_dataset
            dataset_config = VasistDatasetConfig(**config["dataset"])

            train_dataset, valid_dataset, test_dataset = load_vasist_dataset(
                file_path=expand_env_variables_in_path(dataset_config.file_path),
                n_train_samples=dataset_config.n_train_samples,
                n_valid_samples=dataset_config.n_valid_samples,
                n_test_samples=dataset_config.n_test_samples,
                theta_scaler=get_theta_scaler(config.get("theta_scaler", {})),
            )
        case "inaf":
            from fm4ar.datasets.inaf import INAFDatasetConfig, load_inaf_dataset
            dataset_config = INAFDatasetConfig(**config["dataset"])
            train_dataset, valid_dataset, test_dataset = load_inaf_dataset(
                data_dir=expand_env_variables_in_path(dataset_config.data_dir),
                limit=dataset_config.limit,
                verbose=dataset_config.verbose,
                theta_scaler=get_theta_scaler(config.get("theta_scaler", {})),
                auxiliary_data_scaler=get_auxiliary_data_scaler(config.get("auxiliary_data_scaler", {})),
            )
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    return train_dataset, valid_dataset, test_dataset
