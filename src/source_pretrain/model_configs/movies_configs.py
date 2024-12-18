from pathlib import Path

from .ModelConfig import (
    MetricConfig,
    ModelConfig,
    ParameterDistribution,
    ParametersConfig,
    TrainConfig,
    TuneConfig,
)

train_movies_config = ModelConfig(
    dataset_path=Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"),
    epochs=1000,
    early_stopping_patience=5,
    early_stopping_criterion="val_loss",
    val_metric="auc",
    seed=0,
    train_config=TrainConfig(
        n_factors=100,
        learning_rate=0.0003,
        weight_decay=0.07,
        batch_size=512,
        model_save_path=Path("./source_models/best_src_movies.pth"),
    ),
    tune_config=None,
)

tune_movies_config = ModelConfig(
    **{k: v for k, v in train_movies_config.__dict__.items() if k != "tune_config"},
    tune_config=TuneConfig(
        method="bayes",
        metric=MetricConfig(goal="minimize", name="Best val loss"),
        parameters=ParametersConfig(
            n_factors_range=[1, 5, 10, 25, 50, 100],
            learning_rate=ParameterDistribution(
                min=1e-5, max=1e-1, distribution="log_uniform"
            ),
            weight_decay=ParameterDistribution(
                min=1e-6, max=1e-1, distribution="log_uniform"
            ),
            batch_size_range=[64, 128, 256, 512],
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    )
)
