from pathlib import Path
from typing import Literal

from src.model_configs.CommonConfigs import MetricConfig, ParameterDistribution
from src.model_configs.ltn.ModelConfigLtn import ParametersConfigLtn, TuneConfigLtn
from src.model_configs.mf.ModelConfigMf import ParametersConfig, TuneConfigMf

Domains_Type = Literal["books", "movies", "music"]

dataset_name_to_path = {
    "movies": Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"),
    "music": Path("./data/ratings/reviews_CDs_and_Vinyl_5.csv.7z"),
    "books": Path("./data/ratings/reviews_Books_5.csv.7z"),
}

dataset_path_to_name = {path: name for name, path in dataset_name_to_path.items()}

dataset_pair_to_paths_file = {
    "books": {"movies": Path("./data/kg_paths/books-movies.jsonl.7z.001")},
    "movies": {"music": Path("./data/kg_paths/movies-music.jsonl.7z.001")},
    "music": {"movies": Path("./data/kg_paths/music-movies.jsonl.7z")},
}


def get_default_tune_config_mf(
    n_factors_range=(1, 5, 10, 25, 50, 100, 150, 200),
    batch_size_range=(128, 256, 512),
    learning_rate_range=(1e-5, 1e-1),
    weight_decay_range=(1e-6, 1e-1),
) -> TuneConfigMf:
    return TuneConfigMf(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfig(
            n_factors_range=list(n_factors_range),
            learning_rate_range=ParameterDistribution(
                min=learning_rate_range[0], max=learning_rate_range[1], distribution="log_uniform_values"
            ),
            weight_decay_range=ParameterDistribution(
                min=weight_decay_range[0], max=weight_decay_range[1], distribution="log_uniform_values"
            ),
            batch_size_range=list(batch_size_range),
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    )


def get_default_tune_config_ltn_reg(
    n_factors_range=(1, 5, 10, 25, 50, 100, 150, 200),
    batch_size_range=(128, 256, 512),
    learning_rate_range=(1e-5, 1e-1),
    weight_decay_range=(1e-6, 1e-1),
    p_forall_ax1_range=(1, 30),
    p_forall_ax2_range=(1, 30),
    p_sat_agg_range=(1, 30),
) -> TuneConfigLtn:
    return TuneConfigLtn(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigLtn(
            n_factors_range=list(n_factors_range),
            learning_rate_range=ParameterDistribution(
                min=learning_rate_range[0], max=learning_rate_range[1], distribution="log_uniform_values"
            ),
            weight_decay_range=ParameterDistribution(
                min=weight_decay_range[0], max=weight_decay_range[1], distribution="log_uniform_values"
            ),
            batch_size_range=list(batch_size_range),
            p_forall_ax1_range=ParameterDistribution(
                min=p_forall_ax1_range[0], max=p_forall_ax1_range[1], distribution="int_uniform"
            ),
            p_forall_ax2_range=ParameterDistribution(
                min=p_forall_ax2_range[0], max=p_forall_ax2_range[1], distribution="int_uniform"
            ),
            p_sat_agg_range=ParameterDistribution(
                min=p_sat_agg_range[0], max=p_sat_agg_range[1], distribution="int_uniform"
            ),
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    )
