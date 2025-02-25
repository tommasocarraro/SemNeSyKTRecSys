from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.metrics import Valid_Metrics_Type
from src.model_configs.CommonConfigs import DatasetConfig, MetricConfig, ParameterDistribution


@dataclass(frozen=True)
class HyperParamsMf:
    """
    Hyperparameters for MF model
    """

    n_factors: int
    learning_rate: float
    weight_decay: float
    batch_size: int


@dataclass(frozen=True)
class TrainConfigMf:
    """
    Training configuration for MF model
    """

    checkpoint_save_path: Optional[Path]
    final_model_save_path: Optional[Path]
    hyper_params: HyperParamsMf


@dataclass(frozen=True)
class ParametersConfig:
    """
    Hyperparameters' search spaces used in tuning
    """

    n_factors_range: list[int]
    learning_rate_range: ParameterDistribution
    weight_decay_range: ParameterDistribution
    batch_size_range: list[int]


@dataclass(frozen=True)
class TuneConfigMf:
    """
    Wandb tuning configuration for MF model
    """

    method: Literal["bayes"]
    metric: MetricConfig
    parameters: ParametersConfig
    entity_name: str
    exp_name: str
    bayesian_run_count: int
    sweep_id: Optional[str]


@dataclass(frozen=True)
class ModelConfigMf:
    """
    Data class which defines the MF model configuration
    """

    train_dataset_config: DatasetConfig
    other_dataset_config: DatasetConfig
    early_stopping_criterion: Literal["val_loss", "val_metric"]
    val_metric: Valid_Metrics_Type
    train_config: TrainConfigMf
    epochs: int = 1000
    early_stopping_patience: int = 5
    seed: Optional[int] = None
    mf_tune_config: Optional[TuneConfigMf] = None

    def get_train_config_str(self) -> str:
        """
        Gets the string representation of the training configuration
        """
        hyper = self.train_config.hyper_params
        config_str = (
            f"n_factors: {hyper.n_factors}, learning_rate: {hyper.learning_rate}, "
            f"weight_decay: {hyper.weight_decay}, batch_size: {hyper.batch_size}"
        )
        return config_str

    def get_wandb_dict_mf(self):
        """
        Returns the dictionary used for initializing a Wandb sweep
        """
        return {
            "method": self.mf_tune_config.method,
            "metric": {"goal": self.mf_tune_config.metric.goal, "name": self.mf_tune_config.metric.name},
            "parameters": {
                "n_factors": {"values": self.mf_tune_config.parameters.n_factors_range},
                "learning_rate": {
                    "min": self.mf_tune_config.parameters.learning_rate_range.min,
                    "max": self.mf_tune_config.parameters.learning_rate_range.max,
                    "distribution": self.mf_tune_config.parameters.learning_rate_range.distribution,
                },
                "weight_decay": {
                    "min": self.mf_tune_config.parameters.weight_decay_range.min,
                    "max": self.mf_tune_config.parameters.weight_decay_range.max,
                    "distribution": self.mf_tune_config.parameters.weight_decay_range.distribution,
                },
                "batch_size": {"values": self.mf_tune_config.parameters.batch_size_range},
            },
        }
