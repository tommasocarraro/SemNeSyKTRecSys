import os
from pathlib import Path
from typing import Optional

from src.data_preprocessing.Dataset import Domains_Type
from src.data_preprocessing.Split_Strategy import LeaveOneOut
from src.metrics import RankingMetricsType
from src.model_configs.CommonConfigs import DatasetConfig
from src.model_configs.ltn.ModelConfigLtn import ModelConfigLtn, TrainConfigLtn
from src.model_configs.ltn.hyperparams_ltn import get_ltn_hyperparams
from src.model_configs.mf.ModelConfigMf import ModelConfigMf, TrainConfigMf
from src.model_configs.mf.get_config_mf import make_mf_model_paths
from src.model_configs.utils import dataset_name_to_path, dataset_pair_to_paths_file, get_default_tune_config_ltn


def get_config_ltn(
    src_dataset_name: Domains_Type, tgt_dataset_name: Domains_Type, sparsity_sh: float, seed: Optional[int] = None
) -> ModelConfigLtn:
    """
    Retrieves the model configuration for LTN

    :param src_dataset_name: The name of the source dataset
    :param tgt_dataset_name: The name of the target dataset
    :param sparsity_sh: Target domain sparsity factor for shared users
    :param seed: Random seed for reproducibility
    """
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
        src_model_config=ModelConfigMf(
            train_dataset_config=DatasetConfig(
                dataset_name=src_dataset_name,
                dataset_path=dataset_name_to_path[src_dataset_name],
                split_strategy=LeaveOneOut(seed=seed),
                sparsity_sh=sparsity_sh,
            ),
            other_dataset_config=DatasetConfig(
                dataset_name=tgt_dataset_name,
                dataset_path=dataset_name_to_path[tgt_dataset_name],
                split_strategy=LeaveOneOut(seed=seed),
                sparsity_sh=sparsity_sh,
            ),
            early_stopping_criterion="val_metric",
            val_metric=RankingMetricsType.NDCG10,
            train_config=TrainConfigMf(
                hyper_params=source_hyperparams,
                checkpoint_save_path=paths_dict["src_checkpoint"],
                final_model_save_path=paths_dict["src_final_model"],
            ),
            seed=seed,
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
        tgt_train_config=TrainConfigLtn(
            hyper_params=ltn_hyperparams,
            checkpoint_save_path=paths_dict["tgt_checkpoint"],
            final_model_save_path=paths_dict["tgt_final_model"],
        ),
        ltn_reg_tune_config=get_default_tune_config_ltn(),
    )


def make_ltn_model_paths(
    src_dataset_name: Domains_Type, tgt_dataset_name: Domains_Type, sparsity_sh: float, seed: Optional[int] = None
) -> dict[str, Path]:
    """
    Makes the models' paths used by LTN and its source model

    :param src_dataset_name: The name of the source dataset
    :param tgt_dataset_name: The name of the target dataset
    :param sparsity_sh: Target domain sparsity factor for shared users
    :param seed: Random seed for reproducibility
    :return: A dictionary containing the models' paths
    """
    base_path_str = os.path.join(
        "models",
        f"source_domain={src_dataset_name}_target_domain={tgt_dataset_name}_sparsity_sh={sparsity_sh}_seed={seed}",
    )
    checkpoint_path = Path(base_path_str + "_checkpoint.pth")
    final_model_path = Path(base_path_str + "_final_model.pth")
    src_model_paths = make_mf_model_paths(
        train_dataset_name=src_dataset_name, other_dataset_name=tgt_dataset_name, sparsity_sh=1.0
    )
    return {
        "tgt_checkpoint": checkpoint_path,
        "tgt_final_model": final_model_path,
        "src_checkpoint": src_model_paths["checkpoint"],
        "src_final_model": src_model_paths["final_model"],
    }
