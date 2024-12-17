from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class TrainConfig:
    n_factors: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    model_save_path: Path


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
class ModelConfig:
    dataset_path: Path
    epochs: int
    early_stopping_patience: int
    early_stopping_criterion: Literal["val_loss", "val_metric"]
    val_metric: Literal["auc"]
    seed: int
    train_config: TrainConfig
    tune_config: Optional[TuneConfig] = None

    def get_train_config(self):
        return (
            f"n_factors: {self.train_config.n_factors}, learning_rate: {self.train_config.learning_rate}, "
            f"weight_decay: {self.train_config.weight_decay}, batch_size: {self.train_config.batch_size}"
        )

    def get_wandb_dict(self):
        return {
            "method": self.tune_config.method,
            "metric": {
                "goal": self.tune_config.metric.goal,
                "name": self.tune_config.metric.name,
            },
            "parameters": {
                "n_factors": {"values": self.tune_config.parameters.n_factors_range},
                "learning_rate": {
                    "min": self.tune_config.parameters.learning_rate.min,
                    "max": self.tune_config.parameters.learning_rate.max,
                    "distribution": self.tune_config.parameters.learning_rate.distribution,
                },
                "weight_decay": {
                    "min": self.tune_config.parameters.weight_decay.min,
                    "max": self.tune_config.parameters.weight_decay.max,
                    "distribution": self.tune_config.parameters.weight_decay.distribution,
                },
                "batch_size": {"values": self.tune_config.parameters.batch_size_range},
            },
        }
