from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.data_preprocessing.Split_Strategy import SplitStrategy
from src.metrics import Valid_Metrics_Type


@dataclass(frozen=True)
class TrainConfigMf:
    n_factors: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    checkpoint_save_path: Optional[Path]
    final_model_save_path: Optional[Path]


@dataclass(frozen=True)
class TrainConfigLtn(TrainConfigMf):
    p_forall: int


@dataclass(frozen=True)
class TrainConfigLtnReg(TrainConfigLtn):
    top_k_src: int
    p_sat_agg: Optional[int]
    neg_score: Optional[float]


@dataclass(frozen=True)
class MetricConfig:
    goal: Literal["minimize", "maximize"]
    name: str


@dataclass(frozen=True)
class ParameterDistribution:
    min: float
    max: float
    distribution: Literal["log_uniform_values"]


@dataclass(frozen=True)
class ParametersConfigMf:
    n_factors_range: list[int]
    learning_rate: ParameterDistribution
    weight_decay: ParameterDistribution
    batch_size_range: list[int]


@dataclass(frozen=True)
class TuneConfigMf:
    method: Literal["bayes"]
    metric: MetricConfig
    parameters: ParametersConfigMf
    entity_name: str
    exp_name: str
    bayesian_run_count: int
    sweep_id: Optional[str]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_path: Path
    split_strategy: SplitStrategy


@dataclass(frozen=True)
class ParametersConfigLtn:
    p_forall: list[int]


@dataclass(frozen=True)
class ParametersConfigLtnReg:
    top_k_src: list[int]
    p_forall: list[int]
    p_sat_agg: list[int]


@dataclass(frozen=True)
class TuneConfigLtn:
    method: Literal["bayes"]
    metric: MetricConfig
    parameters: ParametersConfigLtn


@dataclass(frozen=True)
class TuneConfigLtnReg:
    method: Literal["bayes"]
    metric: MetricConfig
    parameters: ParametersConfigLtnReg


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
    src_train_config: TrainConfigMf
    ltn_train_config: TrainConfigLtn
    ltn_reg_train_config: TrainConfigLtnReg
    src_mf_tune_config: Optional[TuneConfigMf] = None
    tgt_mf_tune_config: Optional[TuneConfigMf] = None
    ltn_tune_config: Optional[TuneConfigLtn] = None
    ltn_reg_tune_config: Optional[TuneConfigLtnReg] = None

    def get_train_config_str(self, kind: Literal["source", "ltn", "ltn_reg"]) -> str:
        if kind == "source":
            config = self.src_train_config
            config_str = (
                f"n_factors: {config.n_factors}, learning_rate: {config.learning_rate}, "
                f"weight_decay: {config.weight_decay}, batch_size: {config.batch_size}"
            )
        elif kind == "ltn":
            config = self.ltn_train_config
            config_str = (
                f"n_factors: {config.n_factors}, learning_rate: {config.learning_rate}, "
                f"weight_decay: {config.weight_decay}, batch_size: {config.batch_size}, p_forall: {config.p_forall}"
            )
        elif kind == "ltn_reg":
            config = self.ltn_reg_train_config
            config_str = (
                f"n_factors: {config.n_factors}, learning_rate: {config.learning_rate}, "
                f"weight_decay: {config.weight_decay}, batch_size: {config.batch_size}, p_forall: {config.p_forall}, "
                f"top_k_src: {config.top_k_src}, p_forall: {config.p_forall}, "
            )
        else:
            raise ValueError(f"Unknown train kind {kind}")
        return config_str

    def get_wandb_dict_mf(self, kind: Literal["source", "target"]):
        config = self.src_mf_tune_config if kind == "source" else self.tgt_mf_tune_config
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

    def get_wandb_dict_ltn(self):
        config = self.tgt_mf_tune_config
        return {
            "method": "bayes",
            "metric": {"goal": self.ltn_tune_config.metric.goal, "name": self.ltn_tune_config.metric.name},
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
                "p_forall": {"values": self.ltn_tune_config.parameters.p_forall},
            },
        }

    def get_wandb_dict_ltn_reg(self):
        config = self.tgt_mf_tune_config
        return {
            "method": "bayes",
            "metric": {"goal": self.ltn_tune_config.metric.goal, "name": self.ltn_tune_config.metric.name},
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
                "top_k_src": {"values": self.ltn_reg_tune_config.parameters.top_k_src},
                "p_forall": {"values": self.ltn_reg_tune_config.parameters.p_forall},
                "p_sat_agg": {"values": self.ltn_reg_tune_config.parameters.p_sat_agg},
            },
        }
