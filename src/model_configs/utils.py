from os.path import join
from pathlib import Path
from typing import Literal

from loguru import logger

from src.model_configs.ModelConfig import (
    MetricConfig,
    ParameterDistribution,
    ParametersConfigLtnReg,
    ParametersConfigMf,
    TrainConfigLtnReg,
    TrainConfigMf,
    TuneConfigLtnReg,
    TuneConfigMf,
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
        "movies": Path("./data/kg_paths/books(pop:300)->movies(cs:5).json.7z"),
        "music": Path("./data/kg_paths/books(pop:300)->music(cs:5).json.7z"),
    },
    "movies": {
        "books": Path("./data/kg_paths/movies(pop:300)->books(cs:5).json.7z"),
        "music": Path("./data/kg_paths/movies(pop:300)->music(cs:5).json.7z"),
    },
    "music": {
        "books": Path("./data/kg_paths/music(pop:200)->books(cs:5).json.7z"),
        "movies": Path("./data/kg_paths/music(pop:200)->movies(cs:5).json.7z"),
    },
}


def make_train_config_mf(
    src_domain_name: Domains_Type,
    tgt_domain_name: Domains_Type,
    n_factors: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    src_sparsity: float,
    tgt_sparsity: float,
    which_dataset: Literal["source", "target"],
) -> TrainConfigMf:
    check_domains_unequal(src_domain_name, tgt_domain_name)

    return TrainConfigMf(
        n_factors=n_factors,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        checkpoint_save_path=get_checkpoint_weights_path(
            src_domain_name=src_domain_name,
            tgt_domain_name=tgt_domain_name,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            model="mf",
            which_dataset=which_dataset,
        ),
        final_model_save_path=get_best_weights_path(
            src_domain_name=src_domain_name,
            tgt_domain_name=tgt_domain_name,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            model="mf",
            which_dataset=which_dataset,
        ),
    )


def make_train_config_ltn_reg(
    src_domain_name: Domains_Type,
    tgt_domain_name: Domains_Type,
    n_factors: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    p_forall_ax1: int,
    p_forall_ax2: int,
    p_sat_agg: int,
    top_k_src: int,
    neg_score: float,
    src_sparsity: float,
    tgt_sparsity: float,
    which_dataset: Literal["source", "target"],
) -> TrainConfigLtnReg:
    check_domains_unequal(src_domain_name, tgt_domain_name)

    return TrainConfigLtnReg(
        n_factors=n_factors,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        p_forall=p_forall_ax1,
        p_forall_ax2=p_forall_ax2,
        p_sat_agg=p_sat_agg,
        top_k_src=top_k_src,
        neg_score=neg_score,
        checkpoint_save_path=get_checkpoint_weights_path(
            src_domain_name=src_domain_name,
            tgt_domain_name=tgt_domain_name,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            model="ltn_reg",
            which_dataset=which_dataset,
        ),
        final_model_save_path=get_best_weights_path(
            src_domain_name=src_domain_name,
            tgt_domain_name=tgt_domain_name,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            model="ltn_reg",
            which_dataset=which_dataset,
        ),
    )


def get_default_tune_config_mf(
    n_factors_range=(1, 5, 10, 25, 50, 100, 150, 200),
    batch_size_range=(128, 256, 512),
    learning_rate_range=(1e-5, 1e-1),
    weight_decay_range=(1e-6, 1e-1),
) -> TuneConfigMf:
    return TuneConfigMf(
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
) -> TuneConfigLtnReg:
    return TuneConfigLtnReg(
        method="bayes",
        metric=MetricConfig(goal="maximize", name="Best val metric"),
        parameters=ParametersConfigLtnReg(
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


def check_domains_unequal(src_domain_name: Domains_Type, tgt_domain_name: Domains_Type) -> None:
    if src_domain_name == tgt_domain_name:
        logger.error(f"Source domain {src_domain_name} and target domain {tgt_domain_name} are the same")
        exit(1)


def get_best_weights_path(
    src_domain_name: Domains_Type,
    tgt_domain_name: Domains_Type,
    src_sparsity: float,
    tgt_sparsity: float,
    model: Literal["mf", "ltn_reg"],
    which_dataset: Literal["source", "target"],
) -> Path:
    return Path(
        join(
            "source_models",
            f"best_{model}_{src_domain_name}@{src_sparsity}_{tgt_domain_name}@{tgt_sparsity}_{which_dataset}.pth",
        )
    )


def get_checkpoint_weights_path(
    src_domain_name: Domains_Type,
    tgt_domain_name: Domains_Type,
    src_sparsity: float,
    tgt_sparsity: float,
    model: Literal["mf", "ltn_reg"],
    which_dataset: Literal["source", "target"],
) -> Path:
    return Path(
        join(
            "source_models",
            f"checkpoint_{model}_{src_domain_name}@{src_sparsity}_{tgt_domain_name}@{tgt_sparsity}_{which_dataset}.pth",
        )
    )
