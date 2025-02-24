import os.path
from pathlib import Path
from typing import Optional

from .ModelConfigMf import DatasetConfig, ModelConfigMf, TrainConfigMf
from .hyperparams_mf import get_mf_hyperparams
from ..utils import Domains_Type, dataset_name_to_path, get_default_tune_config_mf
from ...data_preprocessing.Split_Strategy import LeaveOneOut
from ...metrics import RankingMetricsType


def get_config_mf(
    train_dataset_name: Domains_Type, other_dataset_name: Domains_Type, sparsity_sh: float, seed: Optional[int] = None
) -> ModelConfigMf:
    paths_dict = make_mf_model_paths(
        train_dataset_name=train_dataset_name, other_dataset_name=other_dataset_name, sparsity_sh=sparsity_sh, seed=seed
    )
    return ModelConfigMf(
        train_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[train_dataset_name],
            split_strategy=LeaveOneOut(seed=seed),
            sparsity_sh=sparsity_sh,
            dataset_name=train_dataset_name,
        ),
        other_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[other_dataset_name],
            split_strategy=LeaveOneOut(seed=seed),
            sparsity_sh=sparsity_sh,
            dataset_name=other_dataset_name,
        ),
        early_stopping_criterion="val_metric",
        val_metric=RankingMetricsType.NDCG10,
        train_config=TrainConfigMf(
            hyper_params=get_mf_hyperparams(domain=train_dataset_name),
            checkpoint_save_path=paths_dict["checkpoint"],
            final_model_save_path=paths_dict["final_model"],
        ),
        mf_tune_config=get_default_tune_config_mf(),
        seed=seed,
    )


def make_mf_model_paths(
    train_dataset_name: Domains_Type, other_dataset_name: Domains_Type, sparsity_sh: float, seed: Optional[int] = None
) -> dict[str, Path]:
    base_path_str = os.path.join(
        "single_domain_models",
        f"train_dataset={train_dataset_name}_other_dataset={other_dataset_name}_sparsity_sh={sparsity_sh}_seed={seed}",
    )
    checkpoint_path = Path(base_path_str + "_checkpoint.pth")
    final_model_path = Path(base_path_str + "_final_model.pth")
    return {"checkpoint": checkpoint_path, "final_model": final_model_path}
