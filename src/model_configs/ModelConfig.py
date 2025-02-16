from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.data_preprocessing.Split_Strategy import SplitStrategy
from src.metrics import Valid_Metrics_Type


@dataclass(frozen=True)
class MfHyperParams:
    n_factors: int
    learning_rate: float
    weight_decay: float
    batch_size: int


@dataclass(frozen=True)
class LtnRegHyperParams(MfHyperParams):
    p_forall_ax1: int
    p_forall_ax2: int
    p_sat_agg: int


@dataclass(frozen=True)
class TrainConfig:
    checkpoint_save_path: Optional[Path]
    final_model_save_path: Optional[Path]


@dataclass(frozen=True)
class MfTrainConfig(TrainConfig):
    hyper_params: MfHyperParams


@dataclass(frozen=True)
class LtnRegTrainConfig(TrainConfig):
    hyper_params: LtnRegHyperParams


@dataclass(frozen=True)
class MetricConfig:
    goal: Literal["minimize", "maximize"]
    name: Literal["Best val loss", "Best val metric"]


@dataclass(frozen=True)
class ParameterDistribution:
    min: float
    max: float
    distribution: Literal["log_uniform_values", "int_uniform"]


@dataclass(frozen=True)
class ParametersConfigMf:
    n_factors_range: list[int]
    learning_rate_range: ParameterDistribution
    weight_decay_range: ParameterDistribution
    batch_size_range: list[int]


@dataclass(frozen=True)
class MfTuneConfig:
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
    user_level: bool
    sparsity: float


@dataclass(frozen=True)
class LtnRegParametersConfig(ParametersConfigMf):
    p_forall_ax1_range: ParameterDistribution
    p_forall_ax2_range: ParameterDistribution
    p_sat_agg_range: ParameterDistribution


@dataclass(frozen=True)
class LtnRegTuneConfig:
    method: Literal["bayes"]
    metric: MetricConfig
    parameters: LtnRegParametersConfig
    entity_name: str
    exp_name: str
    bayesian_run_count: int
    sweep_id: Optional[str]


@dataclass(frozen=True)
class ModelConfig:
    src_dataset_config: DatasetConfig
    tgt_dataset_config: DatasetConfig
    paths_file_path: Path
    early_stopping_criterion: Literal["val_loss", "val_metric"]
    val_metric: Valid_Metrics_Type
    mf_train_config: MfTrainConfig
    ltn_reg_train_config: LtnRegTrainConfig
    epochs: int = 1000
    early_stopping_patience: int = 5
    seed: Optional[int] = None
    mf_tune_config: Optional[MfTuneConfig] = None
    ltn_reg_tune_config: Optional[LtnRegTuneConfig] = None

    def get_train_config_str(self, kind: Literal["mf", "ltn_reg"]) -> str:
        if kind == "mf":
            config = self.mf_train_config
            hyper = config.hyper_params
            config_str = (
                f"n_factors: {hyper.n_factors}, learning_rate: {hyper.learning_rate}, "
                f"weight_decay: {hyper.weight_decay}, batch_size: {hyper.batch_size}"
            )
        elif kind == "ltn_reg":
            config = self.ltn_reg_train_config
            hyper = config.hyper_params
            config_str = (
                f"n_factors: {hyper.n_factors}, learning_rate: {hyper.learning_rate}, "
                f"weight_decay: {hyper.weight_decay}, batch_size: {hyper.batch_size}, p_forall: {hyper.p_forall_ax1}, "
                f"p_forall_ax1: {hyper.p_forall_ax1}, p_forall_ax2: {hyper.p_forall_ax2} "
            )
        else:
            raise ValueError(f"Unknown train kind {kind}")
        return config_str

    def get_wandb_dict_mf(self):
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

    def get_wandb_dict_ltn_reg(self):
        config = self.ltn_reg_tune_config
        return {
            "method": "bayes",
            "metric": {"goal": self.ltn_reg_tune_config.metric.goal, "name": self.ltn_reg_tune_config.metric.name},
            "parameters": {
                "n_factors": {"values": config.parameters.n_factors_range},
                "learning_rate": {
                    "min": config.parameters.learning_rate_range.min,
                    "max": config.parameters.learning_rate_range.max,
                    "distribution": config.parameters.learning_rate_range.distribution,
                },
                "weight_decay": {
                    "min": config.parameters.weight_decay_range.min,
                    "max": config.parameters.weight_decay_range.max,
                    "distribution": config.parameters.weight_decay_range.distribution,
                },
                "batch_size": {"values": config.parameters.batch_size_range},
                "p_forall_ax1": {
                    "min": config.parameters.p_forall_ax1_range.min,
                    "max": config.parameters.p_forall_ax1_range.max,
                    "distribution": config.parameters.p_forall_ax1_range.distribution,
                },
                "p_forall_ax2": {
                    "min": config.parameters.p_forall_ax2_range.min,
                    "max": config.parameters.p_forall_ax2_range.max,
                    "distribution": config.parameters.p_forall_ax2_range.distribution,
                },
                "p_sat_agg": {
                    "min": config.parameters.p_sat_agg_range.min,
                    "max": config.parameters.p_sat_agg_range.max,
                    "distribution": config.parameters.p_sat_agg_range.distribution,
                },
            },
        }
