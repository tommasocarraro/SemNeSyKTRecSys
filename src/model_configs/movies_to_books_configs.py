from src.data_preprocessing.Split_Strategy import LeaveOneOut
from src.metrics import RankingMetricsType
from .ModelConfig import DatasetConfig, ModelConfig
from .utils import (
    Domains_Type,
    dataset_name_to_path,
    dataset_pair_to_paths_file,
    get_default_tune_config_ltn,
    get_default_tune_config_ltn_reg,
    get_default_tune_config_mf,
    make_train_config_ltn,
    make_train_config_ltn_reg,
    make_train_config_mf,
)


_seed = 0
_src_domain: Domains_Type = "movies"
_tgt_domain: Domains_Type = "books"

train_movies_to_books_config = ModelConfig(
    src_dataset_config=DatasetConfig(
        dataset_path=dataset_name_to_path[_src_domain], split_strategy=LeaveOneOut(seed=_seed)
    ),
    tgt_dataset_config=DatasetConfig(
        dataset_path=dataset_name_to_path[_tgt_domain], split_strategy=LeaveOneOut(seed=_seed)
    ),
    paths_file_path=dataset_pair_to_paths_file[_src_domain][_tgt_domain],
    early_stopping_criterion="val_metric",
    val_metric=RankingMetricsType.NDCG10,
    seed=_seed,
    src_train_config=make_train_config_mf(
        src_domain_name=_src_domain,
        tgt_domain_name=_tgt_domain,
        n_factors=200,
        learning_rate=0.00019604,
        weight_decay=0.079963,
        batch_size=512,
    ),
    ltn_train_config=make_train_config_ltn(
        src_domain_name=_src_domain,
        tgt_domain_name=_tgt_domain,
        n_factors=10,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=256,
        p_forall=2,
    ),
    ltn_reg_train_config=make_train_config_ltn_reg(
        src_domain_name=_src_domain,
        tgt_domain_name=_tgt_domain,
        n_factors=10,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=256,
        top_k_src=10,
        p_forall=2,
        p_sat_agg=2,
        neg_score=0.0,
    ),
)

tune_movies_to_books_config = ModelConfig(
    **{k: v for k, v in train_movies_to_books_config.__dict__.items() if not "tune" in k},
    src_mf_tune_config=get_default_tune_config_mf(),
    ltn_tune_config=get_default_tune_config_ltn(),
    ltn_reg_tune_config=get_default_tune_config_ltn_reg()
)
