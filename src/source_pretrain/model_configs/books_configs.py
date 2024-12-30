from math import log
from pathlib import Path

from .ModelConfig import (
    MetricConfig,
    ModelConfig,
    ParameterDistribution,
    ParametersConfig,
    TrainConfig,
    TuneConfig,
)

train_books_config = ModelConfig(
    src_dataset_path=Path("./data/ratings/reviews_Books_5.csv.7z"),
    tgt_dataset_path=Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"),
    paths_file_path=Path("./data/kg_paths/books(pop:300)->movies(cs:5).json.7z"),
    epochs=1000,
    early_stopping_patience=5,
    early_stopping_criterion="val_metric",
    val_metric="NDCG",
    seed=0,
    train_config=TrainConfig(
        n_factors=10,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=256,
        model_save_paths=(
            Path("./source_models/checkpoint_src_movies.pth"),
            Path("./source_models/best_src_movies.pth"),
        ),
    ),
    tune_config=None,
)

tune_books_config = ModelConfig(
    **{k: v for k, v in train_books_config.__dict__.items() if k != "tune_config"},
    tune_config=TuneConfig(
        method="bayes",
        metric=MetricConfig(goal="minimize", name="Best val loss"),
        parameters=ParametersConfig(
            n_factors_range=[100],
            learning_rate=ParameterDistribution(
                min=log(1e-5), max=log(1e-1), distribution="log_uniform"
            ),
            weight_decay=ParameterDistribution(
                min=log(1e-6), max=log(1e-1), distribution="log_uniform"
            ),
            batch_size_range=[512],
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    )
)
