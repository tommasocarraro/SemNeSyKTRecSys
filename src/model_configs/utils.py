import os
from os.path import join
from pathlib import Path
from typing import Literal

from loguru import logger

from src.model_configs.ModelConfig import (
    LtnRegParametersConfig,
    LtnRegTuneConfig,
    MetricConfig,
    MfTuneConfig,
    ParameterDistribution,
    ParametersConfigMf,
)

Domains_Type = Literal["books", "movies", "music"]

dataset_name_to_path = {
    "movies": Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"),
    "music": Path("./data/ratings/reviews_CDs_and_Vinyl_5.csv.7z"),
    "books": Path("./data/ratings/reviews_Books_5.csv.7z"),
}

dataset_path_to_name = {path: name for name, path in dataset_name_to_path.items()}

dataset_pair_to_paths_file = {
    "books": {
        "movies": Path("./data/kg_paths/books-movies.json.7z.001"),
    },
    "movies": {
        "music": Path("./data/kg_paths/movies-music.json.7z.001"),
    },
    "music": {
        "movies": Path("./data/kg_paths/music-movies.json.7z.001"),
    },
}


def get_default_tune_config_mf(
    n_factors_range=(1, 5, 10, 25, 50, 100, 150, 200),
    batch_size_range=(128, 256, 512),
    learning_rate_range=(1e-5, 1e-1),
    weight_decay_range=(1e-6, 1e-1),
) -> MfTuneConfig:
    return MfTuneConfig(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigMf(
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
    p_forall_ax1_range=(1, 2, 5, 10),
    p_forall_ax2_range=(1, 2, 5, 10),
    p_sat_agg_range=(1, 2, 5, 10),
    neg_score_range=(0.1, 5.0),
    top_k_src_range=(10, 50, 100, 200),
) -> LtnRegTuneConfig:
    return LtnRegTuneConfig(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=LtnRegParametersConfig(
            n_factors_range=list(n_factors_range),
            learning_rate_range=ParameterDistribution(
                min=learning_rate_range[0], max=learning_rate_range[1], distribution="log_uniform_values"
            ),
            weight_decay_range=ParameterDistribution(
                min=weight_decay_range[0], max=weight_decay_range[1], distribution="log_uniform_values"
            ),
            batch_size_range=list(batch_size_range),
            p_forall_ax1_range=list(p_forall_ax1_range),
            p_forall_ax2_range=list(p_forall_ax2_range),
            p_sat_agg_range=list(p_sat_agg_range),
            top_k_src_range=list(top_k_src_range),
            neg_score_range=ParameterDistribution(
                min=neg_score_range[0], max=neg_score_range[1], distribution="log_uniform_values"
            ),
        ),
        entity_name="bmxitalia",
        exp_name="amazon",
        bayesian_run_count=50,
        sweep_id=None,
    )


def get_best_weights_path(
    src_domain_name: Domains_Type,
    tgt_domain_name: Domains_Type,
    src_sparsity: float,
    user_level_src: bool,
    tgt_sparsity: float,
    user_level_tgt: bool,
    model: Literal["mf", "ltn_reg"],
    which_dataset: Literal["source", "target"],
) -> Path:
    return Path(
        join(
            "source_models",
            f"best_{model}_{src_domain_name}@{src_sparsity}_ul={user_level_src}_{tgt_domain_name}@{tgt_sparsity}"
            f"_ul={user_level_tgt}_{which_dataset}.pth",
        )
    )


def get_best_weights_path_mf_source(src_domain_name: Domains_Type, src_sparsity: float, user_level_src: bool):
    models_folder = Path("source_models")
    for file_name in os.listdir(models_folder):
        if (
            file_name.startswith(f"best_mf_{src_domain_name}@{src_sparsity}_ul={user_level_src}")
            and file_name.endswith("source.pth")
        ) or (
            file_name.startswith("best_mf_")
            and file_name.endswith(f"{src_domain_name}@{src_sparsity}_ul={user_level_src}_target.pth")
        ):
            return models_folder / Path(file_name)
    logger.error(
        f"Could not find a MF model for source domain: {src_domain_name} with sparsity: {src_sparsity}"
        + (" at user level" if user_level_src else "")
    )
    exit(1)


def get_checkpoint_weights_path(
    src_domain_name: Domains_Type,
    tgt_domain_name: Domains_Type,
    src_sparsity: float,
    user_level_src: bool,
    tgt_sparsity: float,
    user_level_tgt: bool,
    model: Literal["mf", "ltn_reg"],
    which_dataset: Literal["source", "target"],
) -> Path:
    return Path(
        join(
            "source_models",
            f"checkpoint_{model}_{src_domain_name}@{src_sparsity}_ul={user_level_src}_{tgt_domain_name}@{tgt_sparsity}"
            f"_ul={user_level_tgt}_{which_dataset}.pth",
        )
    )
