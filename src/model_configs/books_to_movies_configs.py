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
    TrainConfigLtnReg,
    TrainConfigMf,
    TrainConfigLtn,
    TuneConfigLtn,
    TuneConfigLtnReg,
    TuneConfigMf,
)

_seed = 0

train_books_to_movies_config = ModelConfig(
    src_dataset_config=DatasetConfig(
        dataset_path=Path("./data/ratings/reviews_Books_5.csv.7z"), split_strategy=LeaveOneOut(seed=_seed)
    ),
    tgt_dataset_config=DatasetConfig(
        dataset_path=Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"), split_strategy=LeaveOneOut(seed=_seed)
    ),
    paths_file_path=Path("./data/kg_paths/books(pop:300)->movies(cs:5).json.7z"),
    epochs=1000,
    early_stopping_patience=5,
    early_stopping_criterion="val_metric",
    val_metric=RankingMetricsType.NDCG10,
    seed=_seed,
    src_train_config=TrainConfigMf(
        n_factors=10,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=512,
        checkpoint_save_path=Path("./source_models/checkpoint_src_books_movies.pth"),
        final_model_save_path=Path("./source_models/best_src_books_movies.pth"),
    ),
    ltn_train_config=TrainConfigLtn(
        n_factors=10,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=256,
        checkpoint_save_path=Path("./target_models/checkpoint_ltn_books_movies.pth"),
        final_model_save_path=Path("./target_models/best_ltn_books_movies.pth"),
        p_forall=2,
    ),
    ltn_reg_train_config=TrainConfigLtnReg(
        n_factors=10,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=256,
        checkpoint_save_path=Path("./target_models/checkpoint_ltn_reg_books_movies.pth"),
        final_model_save_path=Path("./target_models/best_ltn_reg_books_movies.pth"),
        top_k_src=10,
        p_forall=2,
        p_sat_agg=2,
        neg_score=0.0,
    ),
)

tune_books_to_movies_config = ModelConfig(
    **{k: v for k, v in train_books_to_movies_config.__dict__.items() if not "tune" in k},
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
