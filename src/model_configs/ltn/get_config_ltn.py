import os
from pathlib import Path
from typing import Optional

from src.data_preprocessing.Split_Strategy import LeaveOneOut
from src.metrics import RankingMetricsType
from src.model_configs.CommonConfigs import DatasetConfig
from src.model_configs.ltn.ModelConfigLtn import ModelConfigLtn, TrainConfigLtn
from src.model_configs.ltn.hyperparams_ltn import get_ltn_hyperparams
from src.model_configs.utils import (
    Domains_Type,
    dataset_name_to_path,
    dataset_pair_to_paths_file,
    get_default_tune_config_ltn_reg,
)


def get_config_ltn(
    src_dataset_name: Domains_Type, tgt_dataset_name: Domains_Type, sparsity_sh: float, seed: Optional[int] = None
) -> ModelConfigLtn:
    ltn_hyperparams, source_hyperparams = get_ltn_hyperparams(
        src_domain=src_dataset_name, tgt_domain=tgt_dataset_name, sparsity_sh=sparsity_sh
    )
    paths_dict = make_ltn_model_paths(
        src_dataset_name=src_dataset_name, tgt_dataset_name=tgt_dataset_name, sparsity_sh=sparsity_sh, seed=seed
    )
    return ModelConfigLtn(
        src_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[src_dataset_name],
            split_strategy=LeaveOneOut(seed=seed),
            sparsity_sh=sparsity_sh,
            dataset_name=src_dataset_name,
        ),
        tgt_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[tgt_dataset_name],
            split_strategy=LeaveOneOut(seed=seed),
            sparsity_sh=sparsity_sh,
            dataset_name=tgt_dataset_name,
        ),
        paths_file_path=dataset_pair_to_paths_file[src_dataset_name][tgt_dataset_name],
        early_stopping_criterion="val_metric",
        val_metric=RankingMetricsType.NDCG10,
        train_config=TrainConfigLtn(
            hyper_params_source=source_hyperparams,
            hyper_params_target=ltn_hyperparams,
            checkpoint_save_path=paths_dict["checkpoint"],
            final_model_save_path=paths_dict["final_model"],
        ),
        ltn_reg_tune_config=get_default_tune_config_ltn_reg(),
    )


def make_ltn_model_paths(
    src_dataset_name: Domains_Type, tgt_dataset_name: Domains_Type, sparsity_sh: float, seed: Optional[int] = None
) -> dict[str, Path]:
    base_path_str = os.path.join(
        "domain_pairs_models",
        f"source_domain={src_dataset_name}_target_domain={tgt_dataset_name}_sparsity_sh={sparsity_sh}_seed={seed}",
    )
    checkpoint_path = Path(base_path_str + "_checkpoint.pth")
    final_model_path = Path(base_path_str + "_final_model.pth")
    return {"checkpoint": checkpoint_path, "final_model": final_model_path}
