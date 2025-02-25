from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.data_preprocessing.Dataset import Domains_Type
from src.data_preprocessing.Split_Strategy import SplitStrategy


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset configuration class
    """

    dataset_name: Domains_Type
    dataset_path: Path
    split_strategy: SplitStrategy
    sparsity_sh: float


@dataclass(frozen=True)
class MetricConfig:
    """
    Metric used by wandb for tuning
    """

    goal: Literal["minimize", "maximize"]
    name: Literal["Best val loss", "Best val metric"]


@dataclass(frozen=True)
class ParameterDistribution:
    """
    Parameter distribution used by wandb for tuning
    """

    min: float
    max: float
    distribution: Literal["log_uniform_values", "int_uniform"]
