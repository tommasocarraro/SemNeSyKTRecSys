from math import log
from pathlib import Path

from src.data_preprocessing.Split_Strategy import LeaveOneOut
from src.metrics import RankingMetricsType
from .ModelConfig import (
    DatasetConfig,
    MetricConfig,
    ModelConfig,
    ParameterDistribution,
    ParametersConfigLtn,
    ParametersConfigLtnReg,
    ParametersConfigMf,
    TrainConfigSrc,
    TrainConfigTgt,
    TuneConfigLtn,
    TuneConfigLtnReg,
    TuneConfigMf,
)

_seed = 0

train_movies_to_books_config = ModelConfig(
    src_dataset_config=DatasetConfig(
        dataset_path=Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"), split_strategy=LeaveOneOut(seed=_seed)
    ),
    tgt_dataset_config=DatasetConfig(
        dataset_path=Path("./data/ratings/reviews_Books_5.csv.7z"), split_strategy=LeaveOneOut(seed=_seed)
    ),
    paths_file_path=Path("./data/kg_paths/movies(pop:300)->books(cs:5).json.7z"),
    epochs=1000,
    early_stopping_patience=5,
    early_stopping_criterion="val_metric",
    val_metric=RankingMetricsType.NDCG10,
    seed=_seed,
    src_train_config=TrainConfigSrc(
        n_factors=100,
        learning_rate=0.0003,
        weight_decay=0.07,
        batch_size=512,
        checkpoint_save_path=Path("./source_models/checkpoint_src_movies_books.pth"),
        final_model_save_path=Path("./source_models/best_src_movies_books.pth"),
    ),
    tgt_train_config=TrainConfigTgt(
        n_factors=100,
        learning_rate=0.0003,
        weight_decay=0.07,
        batch_size=512,
        checkpoint_save_path=Path("./target_models/checkpoint_tgt_movies_books.pth"),
        final_model_save_path=Path("./target_models/best_tgt_movies_books.pth"),
        top_k_src=10,
        p_forall=2,
        p_sat_agg=2,
        neg_score=0.0,
    ),
)

tune_movies_books_config = ModelConfig(
    **{k: v for k, v in train_movies_to_books_config.__dict__.items() if not "tune" in k},
    src_mf_tune_config=TuneConfigMf(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigMf(
            n_factors_range=[1, 5, 10, 25, 50, 100, 150, 200],
            learning_rate=ParameterDistribution(min=1e-5, max=1e-1, distribution="log_uniform_values"),
            weight_decay=ParameterDistribution(min=1e-6, max=1e-1, distribution="log_uniform_values"),
            batch_size_range=[128, 256, 512],
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    ),
    tgt_mf_tune_config=TuneConfigMf(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigMf(
            n_factors_range=[1, 5, 10, 25, 50, 100, 150, 200],
            learning_rate=ParameterDistribution(min=1e-5, max=1e-1, distribution="log_uniform_values"),
            weight_decay=ParameterDistribution(min=1e-6, max=1e-1, distribution="log_uniform_values"),
            batch_size_range=[128, 256, 512],
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    ),
    ltn_tune_config=TuneConfigLtn(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigLtn(p_forall=[1, 2, 5, 10]),
    ),
    ltn_reg_tune_config=TuneConfigLtnReg(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigLtnReg(
            p_forall=[1, 2, 5, 10], p_sat_agg=[1, 2, 5, 10], top_k_src=[10, 50, 100, 200]
        ),
    )
)
