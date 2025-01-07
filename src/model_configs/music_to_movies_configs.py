from math import log
from pathlib import Path

from src.data_preprocessing.Split_Strategy import LeaveLastOut
from src.metrics import RankingMetricsType
from .ModelConfig import (
    DatasetConfig,
    MetricConfig,
    ModelConfig,
    ParameterDistribution,
    ParametersConfig,
    TrainConfig,
    TuneConfig,
)

_seed = 0

train_music_to_movies_config = ModelConfig(
    src_dataset_config=DatasetConfig(
        dataset_path=Path("./data/ratings/reviews_CDs_and_Vinyl_5.csv.7z"), split_strategy=LeaveLastOut(seed=_seed)
    ),
    tgt_dataset_config=DatasetConfig(
        dataset_path=Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"), split_strategy=LeaveLastOut(seed=_seed)
    ),
    paths_file_path=Path("./data/kg_paths/music(pop:200)->movies(cs:5).json.7z"),
    epochs=1000,
    early_stopping_patience=5,
    early_stopping_criterion="val_metric",
    val_metric=RankingMetricsType.NDCG,
    seed=_seed,
    src_train_config=TrainConfig(
        n_factors=100,
        learning_rate=0.0001,
        weight_decay=0.0001,
        batch_size=512,
        model_save_paths=(
            Path("./source_models/checkpoint_src_music_movies.pth"),
            Path("./source_models/best_src_music_movies.pth"),
        ),
    ),
    tgt_train_config=TrainConfig(
        n_factors=100,
        learning_rate=0.0001,
        weight_decay=0.0001,
        batch_size=512,
        model_save_paths=(
            Path("./target_models/checkpoint_tgt_music_movies.pth"),
            Path("./target_models/best_tgt_music_movies.pth"),
        ),
    ),
)

tune_music_to_movies_config = ModelConfig(
    **{k: v for k, v in train_music_to_movies_config.__dict__.items() if not "tune" in k},
    src_tune_config=TuneConfig(
        method="bayes",
        metric=MetricConfig(goal="minimize", name="Best val loss"),
        parameters=ParametersConfig(
            n_factors_range=[1, 5, 10, 25, 50, 100, 150, 200],
            learning_rate=ParameterDistribution(min=log(1e-5), max=log(1e-1), distribution="log_uniform"),
            weight_decay=ParameterDistribution(min=log(1e-6), max=log(1e-1), distribution="log_uniform"),
            batch_size_range=[256, 512],
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    ),
    tgt_tune_config=TuneConfig(
        method="bayes",
        metric=MetricConfig(goal="minimize", name="Best val loss"),
        parameters=ParametersConfig(
            n_factors_range=[1, 5, 10, 25, 50, 100, 150, 200],
            learning_rate=ParameterDistribution(min=log(1e-5), max=log(1e-1), distribution="log_uniform"),
            weight_decay=ParameterDistribution(min=log(1e-6), max=log(1e-1), distribution="log_uniform"),
            batch_size_range=[256, 512],
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    ),
)
