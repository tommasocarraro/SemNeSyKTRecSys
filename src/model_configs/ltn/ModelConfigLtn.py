from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.metrics import Valid_Metrics_Type
from src.model_configs.CommonConfigs import DatasetConfig, MetricConfig, ParameterDistribution
from src.model_configs.mf.ModelConfigMf import HyperParamsMf, ModelConfigMf


@dataclass(frozen=True)
class HyperParamsLtn(HyperParamsMf):
    """
    Hyperparameters for LTN model
    """

    p_forall_ax1: int
    p_forall_ax2: int
    p_sat_agg: int


@dataclass(frozen=True)
class TrainConfigLtn:
    """
    Training configuration for LTN model
    """

    checkpoint_save_path: Optional[Path]
    final_model_save_path: Optional[Path]
    hyper_params: HyperParamsLtn


@dataclass(frozen=True)
class ParametersConfigLtn:
    """
    Hyperparameters' search spaces used in tuning
    """

    n_factors_range: list[int]
    learning_rate_range: ParameterDistribution
    weight_decay_range: ParameterDistribution
    batch_size_range: list[int]
    p_forall_ax1_range: ParameterDistribution
    p_forall_ax2_range: ParameterDistribution
    p_sat_agg_range: ParameterDistribution


@dataclass(frozen=True)
class TuneConfigLtn:
    """
    Wandb tuning configuration for LTN model
    """

    method: Literal["bayes"]
    metric: MetricConfig
    parameters: ParametersConfigLtn
    entity_name: str
    exp_name: str
    bayesian_run_count: int
    sweep_id: Optional[str]


@dataclass(frozen=True)
class ModelConfigLtn:
    """
    Data class which defines the LTN model configuration
    """

    src_dataset_config: DatasetConfig
    src_model_config: ModelConfigMf
    tgt_dataset_config: DatasetConfig
    paths_file_path: Path
    early_stopping_criterion: Literal["val_loss", "val_metric"]
    val_metric: Valid_Metrics_Type
    tgt_train_config: TrainConfigLtn
    epochs: int = 1000
    early_stopping_patience: int = 5
    seed: Optional[int] = None
    ltn_reg_tune_config: Optional[TuneConfigLtn] = None

    def get_train_config_str(self) -> str:
        """
        Gets the string representation of the training configuration
        """
        hyper = self.tgt_train_config.hyper_params
        config_str = (
            f"n_factors: {hyper.n_factors}, learning_rate: {hyper.learning_rate}, "
            f"weight_decay: {hyper.weight_decay}, batch_size: {hyper.batch_size}, p_forall_ax: {hyper.p_forall_ax1}, "
            f"p_forall_ax2: {hyper.p_forall_ax2}, p_sat_agg: {hyper.p_sat_agg} "
        )
        return config_str

    def get_wandb_dict_ltn(self):
        """
        Returns the dictionary used for initializing a Wandb sweep
        """
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
