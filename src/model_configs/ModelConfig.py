from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.data_preprocessing.Split_Strategy import SplitStrategy
from src.source_pretrain.metrics import Valid_Metrics_Type


@dataclass(frozen=True)
class TrainConfig:
    n_factors: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    model_save_paths: tuple[Path, Path]


@dataclass(frozen=True)
class MetricConfig:
    goal: Literal["minimize", "maximize"]
    name: str


@dataclass(frozen=True)
class ParameterDistribution:
    min: float
    max: float
    distribution: Literal["log_uniform"]


@dataclass(frozen=True)
class ParametersConfig:
    n_factors_range: list[int]
    learning_rate: ParameterDistribution
    weight_decay: ParameterDistribution
    batch_size_range: list[int]


@dataclass(frozen=True)
class TuneConfig:
    method: Literal["bayes"]
    metric: MetricConfig
    parameters: ParametersConfig
    entity_name: str
    exp_name: str
    bayesian_run_count: int
    sweep_id: Optional[str]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_path: Path
    split_strategy: SplitStrategy


@dataclass(frozen=True)
class ModelConfig:
    src_dataset_config: DatasetConfig
    tgt_dataset_config: DatasetConfig
    paths_file_path: Path
    epochs: int
    early_stopping_patience: int
    early_stopping_criterion: Literal["val_loss", "val_metric"]
    val_metric: Valid_Metrics_Type
    seed: int
    src_train_config: TrainConfig
    tgt_train_config: TrainConfig
    src_tune_config: Optional[TuneConfig] = None
    tgt_tune_config: Optional[TuneConfig] = None

    def get_train_config(self, kind: Literal["source", "target"]):
        config = self.src_train_config if kind == "source" else self.tgt_train_config
        return (
            f"n_factors: {config.n_factors}, learning_rate: {config.learning_rate}, "
            f"weight_decay: {config.weight_decay}, batch_size: {config.batch_size}"
        )

    def get_wandb_dict(self, kind: Literal["source", "target"]):
        config = self.src_tune_config if kind == "source" else self.tgt_tune_config
        return {
            "method": config.method,
            "metric": {"goal": config.metric.goal, "name": config.metric.name},
            "parameters": {
                "n_factors": {"values": config.parameters.n_factors_range},
                "learning_rate": {
                    "min": config.parameters.learning_rate.min,
                    "max": config.parameters.learning_rate.max,
                    "distribution": config.parameters.learning_rate.distribution,
                },
                "weight_decay": {
                    "min": config.parameters.weight_decay.min,
                    "max": config.parameters.weight_decay.max,
                    "distribution": config.parameters.weight_decay.distribution,
                },
                "batch_size": {"values": config.parameters.batch_size_range},
            },
        }
