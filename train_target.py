import argparse
import os
from pathlib import Path
from typing import Literal, Optional

import dotenv
import torch
import wandb
from loguru import logger

from src.cross_domain.ltn_trainer import LTNRegTrainer, LTNTrainer
from src.cross_domain.tuning import ltn_tuning, ltn_tuning_reg
from src.cross_domain.utils import get_reg_axiom_data
from src.data_loader import DataLoader, ValDataLoader
from src.data_preprocessing.Dataset import Dataset
from src.data_preprocessing.process_source_target import process_source_target
from src.model import MatrixFactorization
from src.model_configs import ModelConfig, get_config
from src.pretrain_source.inference import generate_pre_trained_src_matrix
from src.utils import set_seed

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
parser.add_argument(
    "datasets",
    type=str,
    help="Datasets to use (appears after --train or --tune)",
    nargs=2,
    choices=["movies", "music", "books"],
)
parser.add_argument("--src_model_path", type=str, help="Path to pretrained source model", required=False)
parser.add_argument("--clear", help="recompute dataset", action="store_true")


def main():
    args = parser.parse_args()

    src_dataset_name, tgt_dataset_name = args.datasets
    src_model_path = args.src_model_path
    kind: Literal["train", "tune"] = "train" if args.train else "tune"

    config = get_config(src_dataset_name=src_dataset_name, tgt_dataset_name=tgt_dataset_name, kind=kind)
    set_seed(config.seed)

    dataset = process_source_target(
        src_dataset_config=config.src_dataset_config,
        tgt_dataset_config=config.tgt_dataset_config,
        paths_file_path=config.paths_file_path,
        save_dir_path=Path("data/saved_data/"),
        clear_saved_dataset=args.clear,
    )

    src_model_path = Path(src_model_path) if src_model_path is not None else None

    if args.train:
        train_target(dataset, config, src_model_path)
    elif args.tune:
        tune_target(dataset, config, src_model_path)


def train_target(dataset: Dataset, config: ModelConfig, src_model_path: Optional[Path]):
    kind: Literal["ltn", "ltn_reg"] = "ltn" if src_model_path is None else "ltn_reg"
    train_config = config.ltn_train_config if kind == "ltn" else config.ltn_reg_train_config

    logger.info(f"Training the model with configuration: {config.get_train_config_str(kind)}")

    tr_loader = DataLoader(data=dataset.tgt_tr, ui_matrix=dataset.tgt_ui_matrix, batch_size=train_config.batch_size)

    val_loader = ValDataLoader(
        data=dataset.tgt_val, ui_matrix=dataset.tgt_ui_matrix, batch_size=train_config.batch_size
    )

    mf_model_tgt = MatrixFactorization(
        n_users=dataset.tgt_n_users, n_items=dataset.tgt_n_items, n_factors=train_config.n_factors
    )

    if src_model_path is not None:
        mf_model_src = MatrixFactorization(
            n_users=dataset.src_n_users, n_items=dataset.src_n_items, n_factors=config.src_train_config.n_factors
        )

        processed_interactions = get_reg_axiom_data(
            src_ui_matrix=dataset.src_ui_matrix,
            tgt_ui_matrix=dataset.tgt_ui_matrix,
            n_sh_users=dataset.n_sh_users,
            sim_matrix=dataset.sim_matrix,
            top_k_items=generate_pre_trained_src_matrix(
                mf_model=mf_model_src,
                best_weights_path=config.src_train_config.final_model_save_path,
                n_shared_users=dataset.n_sh_users,
                top_k_src=train_config.top_k_src,
                batch_size=train_config.batch_size,
            ),
        )

        tr = LTNRegTrainer(
            mf_model=mf_model_tgt,
            optimizer=torch.optim.AdamW(
                mf_model_tgt.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
            ),
            p_forall=train_config.p_forall,
            p_sat_agg=train_config.p_sat_agg,
            neg_score_value=train_config.neg_score,
            processed_interactions=processed_interactions,
        )
    else:
        tr = LTNTrainer(
            mf_model=mf_model_tgt,
            optimizer=torch.optim.AdamW(
                mf_model_tgt.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
            ),
            p_forall=train_config.p_forall,
        )

    tr.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        early=config.early_stopping_patience,
        verbose=1,
        early_stopping_criterion=config.early_stopping_criterion,
        checkpoint_save_path=train_config.checkpoint_save_path,
        final_model_save_path=train_config.final_model_save_path,
    )

    val_metric_results, _ = tr.validate(val_loader=val_loader, val_metric=config.val_metric, use_val_loss=False)

    logger.info(f"Training complete. Final validation {config.val_metric.name}: {val_metric_results:.4f}")

    te_loader = ValDataLoader(
        data=dataset.src_te, ui_matrix=dataset.src_ui_matrix, batch_size=train_config.batch_size
    )
    te_metric_results, _ = tr.validate(te_loader, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric}: {te_metric_results:.4f}")


def tune_target(dataset: Dataset, config: ModelConfig, src_model_path: Optional[Path]):
    if config.tgt_mf_tune_config is None:
        raise ValueError("Missing tuning configuration")

    kind: Literal["ltn", "ltn_reg"] = "ltn" if src_model_path is None else "ltn_reg"
    train_config = config.ltn_train_config if kind == "ltn" else config.ltn_reg_train_config

    # wandb login
    if not dotenv.load_dotenv():
        logger.error("No environment variables found")
        exit(1)
    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        logger.error("Missing Wandb API key in the environment file")
        exit(1)
    wandb.login(key=api_key)

    if src_model_path is not None:
        mf_model_src = MatrixFactorization(
            n_users=dataset.src_n_users, n_items=dataset.src_n_items, n_factors=config.src_train_config.n_factors
        )
        ltn_tuning_reg(
            seed=config.seed,
            tune_config=config.get_wandb_dict_ltn(),
            train_set=dataset.tgt_tr,
            val_set=dataset.tgt_val,
            val_batch_size=train_config.batch_size,
            n_users=dataset.tgt_n_users,
            n_items=dataset.tgt_n_items,
            src_ui_matrix=dataset.src_ui_matrix,
            tgt_ui_matrix=dataset.tgt_ui_matrix,
            val_metric=config.val_metric,
            n_epochs=config.epochs,
            early=config.early_stopping_patience,
            early_stopping_criterion=config.early_stopping_criterion,
            entity_name=config.tgt_mf_tune_config.entity_name,
            exp_name=config.tgt_mf_tune_config.exp_name,
            bayesian_run_count=config.tgt_mf_tune_config.bayesian_run_count,
            sweep_id=config.tgt_mf_tune_config.sweep_id,
            best_src_model_path=src_model_path,
            src_batch_size=train_config.batch_size,
            sim_matrix=dataset.sim_matrix,
            n_sh_users=dataset.n_sh_users,
            mf_model_src=mf_model_src,
        )
    else:
        ltn_tuning(
            seed=config.seed,
            tune_config=config.get_wandb_dict_ltn(),
            train_set=dataset.tgt_tr,
            val_set=dataset.tgt_val,
            val_batch_size=train_config.batch_size,
            n_users=dataset.tgt_n_users,
            n_items=dataset.tgt_n_items,
            tgt_ui_matrix=dataset.tgt_ui_matrix,
            val_metric=config.val_metric,
            n_epochs=config.epochs,
            early=config.early_stopping_patience,
            early_stopping_criterion=config.early_stopping_criterion,
            entity_name=config.tgt_mf_tune_config.entity_name,
            exp_name=config.tgt_mf_tune_config.exp_name,
            bayesian_run_count=config.tgt_mf_tune_config.bayesian_run_count,
            sweep_id=config.tgt_mf_tune_config.sweep_id,
        )


if __name__ == "__main__":
    main()
