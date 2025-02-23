from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.data_preprocessing.Split_Strategy import SplitStrategy


@dataclass(frozen=True)
class DatasetConfig:
    dataset_path: Path
    split_strategy: SplitStrategy
    sparsity_sh: float


@dataclass(frozen=True)
class MetricConfig:
    goal: Literal["minimize", "maximize"]
    name: Literal["Best val loss", "Best val metric"]


@dataclass(frozen=True)
class ParameterDistribution:
    min: float
    max: float
    distribution: Literal["log_uniform_values", "int_uniform"]
