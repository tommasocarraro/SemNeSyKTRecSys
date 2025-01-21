from typing import Literal

from src.data_preprocessing.Split_Strategy import LeaveOneOut
from src.metrics import RankingMetricsType
from .ModelConfig import DatasetConfig, LtnRegTrainConfig, MfHyperParams, MfTrainConfig, ModelConfig
from .single_domains_hyperparams import mf_movies_hyperparams, mf_music_hyperparams
from .utils import (
    Domains_Type,
    dataset_name_to_path,
    dataset_pair_to_paths_file,
    get_best_weights_path,
    get_checkpoint_weights_path,
    get_default_tune_config_ltn_reg,
    get_default_tune_config_mf,
)

_seed = 0
_src_domain: Domains_Type = "music"
_tgt_domain: Domains_Type = "movies"


def make_music_to_movies_config(
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
        mf_train_config=MfTrainConfig(
            mf_hyper_params=mf_music_hyperparams if which_dataset == "source" else mf_movies_hyperparams,
            checkpoint_save_path=get_checkpoint_weights_path(
                src_domain_name=_src_domain,
                tgt_domain_name=_tgt_domain,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="mf",
                which_dataset=which_dataset,
            ),
            final_model_save_path=get_best_weights_path(
                src_domain_name=_src_domain,
                tgt_domain_name=_tgt_domain,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="mf",
                which_dataset=which_dataset,
            ),
        ),
        ltn_reg_train_config=LtnRegTrainConfig(
            mf_hyper_params=MfHyperParams(n_factors=10, learning_rate=0.001, weight_decay=0.001, batch_size=256),
            p_forall_ax1=2,
            p_forall_ax2=2,
            p_sat_agg=2,
            neg_score=0.0,
            top_k_src=10,
            checkpoint_save_path=get_checkpoint_weights_path(
                src_domain_name=_src_domain,
                tgt_domain_name=_tgt_domain,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="ltn_reg",
                which_dataset=which_dataset,
            ),
            final_model_save_path=get_best_weights_path(
                src_domain_name=_src_domain,
                tgt_domain_name=_tgt_domain,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                model="ltn_reg",
                which_dataset=which_dataset,
            ),
        ),
        mf_tune_config=get_default_tune_config_mf(),
        ltn_reg_tune_config=get_default_tune_config_ltn_reg(),
    )
