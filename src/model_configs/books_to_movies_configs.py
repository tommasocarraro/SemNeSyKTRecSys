from typing import Literal

from src.data_preprocessing.Split_Strategy import LeaveOneOut
from src.metrics import RankingMetricsType
from .ModelConfig import DatasetConfig, ModelConfig
from .utils import (
    Domains_Type,
    dataset_name_to_path,
    dataset_pair_to_paths_file,
    get_default_tune_config_ltn_reg,
    get_default_tune_config_mf,
    make_train_config_ltn_reg,
    make_train_config_mf,
)

_seed = 0
_src_domain: Domains_Type = "books"
_tgt_domain: Domains_Type = "movies"


def make_books_to_movies_config(
    src_sparsity: float, tgt_sparsity: float, which_dataset: Literal["source", "target"]
) -> ModelConfig:
    return ModelConfig(
        src_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[_src_domain],
            split_strategy=LeaveOneOut(seed=_seed),
            sparsity=src_sparsity,
        ),
        tgt_dataset_config=DatasetConfig(
            dataset_path=dataset_name_to_path[_tgt_domain],
            split_strategy=LeaveOneOut(seed=_seed),
            sparsity=tgt_sparsity,
        ),
        paths_file_path=dataset_pair_to_paths_file[_src_domain][_tgt_domain],
        early_stopping_criterion="val_metric",
        val_metric=RankingMetricsType.NDCG10,
        seed=_seed,
        mf_train_config=make_train_config_mf(
            src_domain_name=_src_domain,
            tgt_domain_name=_tgt_domain,
            n_factors=200,
            learning_rate=0.0001810210202954595,
            weight_decay=0.00026616161728996144,
            batch_size=256,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            which_dataset=which_dataset,
        ),
        ltn_reg_train_config=make_train_config_ltn_reg(
            src_domain_name=_src_domain,
            tgt_domain_name=_tgt_domain,
            n_factors=10,
            learning_rate=0.001,
            weight_decay=0.001,
            batch_size=256,
            top_k_src=10,
            p_forall_ax1=2,
            p_forall_ax2=2,
            p_sat_agg=2,
            neg_score=0.0,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            which_dataset=which_dataset,
        ),
        mf_tune_config=get_default_tune_config_mf(),
        ltn_reg_tune_config=get_default_tune_config_ltn_reg(),
    )
