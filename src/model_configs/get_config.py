from typing import Literal, Optional

from .ModelConfig import DatasetConfig, LtnRegTrainConfig, MfTrainConfig, ModelConfig
from .ltn_hyperparams import get_ltn_hyperparams
from .mf_hyperparams import get_mf_hyperparams
from .utils import (
    Domains_Type,
    dataset_name_to_path,
    dataset_pair_to_paths_file,
    get_best_weights_path,
    get_checkpoint_weights_path,
    get_default_tune_config_ltn_reg,
    get_default_tune_config_mf,
)
from ..data_preprocessing.Split_Strategy import LeaveOneOut
from ..metrics import RankingMetricsType


def get_config(
    src_dataset_name: Domains_Type,
    tgt_dataset_name: Domains_Type,
    src_sparsity: float,
    tgt_sparsity: float,
    which_dataset: Literal["source", "target"],
    seed: Optional[int],
) -> ModelConfig:
    return ModelConfig(
        src_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[src_dataset_name],
            split_strategy=LeaveOneOut(seed=seed),
            sparsity=src_sparsity,
        ),
        tgt_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[tgt_dataset_name],
            split_strategy=LeaveOneOut(seed=seed),
            sparsity=tgt_sparsity,
        ),
        paths_file_path=dataset_pair_to_paths_file[src_dataset_name][tgt_dataset_name],
        early_stopping_criterion="val_metric",
        val_metric=RankingMetricsType.NDCG10,
        mf_train_config=MfTrainConfig(
            hyper_params=(
                get_mf_hyperparams(domain=src_dataset_name, sparsity=src_sparsity)
                if which_dataset == "source"
                else get_mf_hyperparams(domain=tgt_dataset_name, sparsity=tgt_sparsity)
            ),
            checkpoint_save_path=get_checkpoint_weights_path(
                src_domain_name=src_dataset_name,
                tgt_domain_name=tgt_dataset_name,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="mf",
                which_dataset=which_dataset,
            ),
            final_model_save_path=get_best_weights_path(
                src_domain_name=src_dataset_name,
                tgt_domain_name=tgt_dataset_name,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="mf",
                which_dataset=which_dataset,
            ),
        ),
        ltn_reg_train_config=LtnRegTrainConfig(
            hyper_params=get_ltn_hyperparams(
                src_domain=src_dataset_name, tgt_domain=tgt_dataset_name, tgt_sparsity=tgt_sparsity
            ),
            checkpoint_save_path=get_checkpoint_weights_path(
                src_domain_name=src_dataset_name,
                tgt_domain_name=tgt_dataset_name,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="ltn_reg",
                which_dataset=which_dataset,
            ),
            final_model_save_path=get_best_weights_path(
                src_domain_name=src_dataset_name,
                tgt_domain_name=tgt_dataset_name,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="ltn_reg",
                which_dataset=which_dataset,
            ),
        ),
        mf_tune_config=get_default_tune_config_mf(),
        ltn_reg_tune_config=get_default_tune_config_ltn_reg(),
    )
