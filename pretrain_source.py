import argparse
import os
from pathlib import Path
from typing import Literal

import dotenv
import torch
import wandb
from loguru import logger

from src.data_preprocessing.Dataset import Dataset
from src.data_preprocessing.process_source_target import process_source_target
from src.model_configs import ModelConfig, get_config
from src.data_loader import DataLoader, ValDataLoader
from src.source.loss import BPRLoss
from src.metrics import RankingMetricsType
from src.model import MatrixFactorization
from src.source.mf_trainer import MfTrainer
from src.source.tuning import mf_tuning
from src.utils import set_seed

parser = argparse.ArgumentParser(description="PyTorch BPR Training")
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
parser.add_argument("--clear", help="recompute dataset", action="store_true")


def main_source():
    args = parser.parse_args()

    src_dataset_name, tgt_dataset_name = args.datasets
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

    if args.train:
        train_source(dataset, config)
    elif args.tune:
        tune_source(dataset, config)


def train_source(dataset: Dataset, config: ModelConfig):
    logger.info(f"Training the model with configuration: {config.get_train_config('source')}")

    tr_loader = DataLoader(
        data=dataset.src_tr, ui_matrix=dataset.src_ui_matrix, batch_size=config.src_train_config.batch_size
    )

    val_loader = ValDataLoader(
        data=dataset.src_val, ui_matrix=dataset.src_ui_matrix, batch_size=config.src_train_config.batch_size
    )

    mf = MatrixFactorization(
        n_users=dataset.src_n_users, n_items=dataset.src_n_items, n_factors=config.src_train_config.n_factors
    )

    tr = MfTrainer(
        model=mf,
        optimizer=torch.optim.AdamW(
            mf.parameters(),
            lr=config.src_train_config.learning_rate,
            weight_decay=config.src_train_config.weight_decay,
        ),
        loss=BPRLoss(),
    )

    tr.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        early=config.early_stopping_patience,
        verbose=1,
        early_stopping_criterion=config.early_stopping_criterion,
        save_paths=config.src_train_config.model_save_paths,
    )

    logger.info(
        f"Training complete. Final validation AUC and loss: {tr.validate(val_loader, RankingMetricsType.NDCG, True)}"
    )

    if dataset.src_te is not None:
        te_loader = ValDataLoader(
            data=dataset.src_te, ui_matrix=dataset.src_ui_matrix, batch_size=config.src_train_config.batch_size
        )
        te_metric, _ = tr.validate(te_loader, val_metric=config.val_metric)
        logger.info(f"Test {config.val_metric}: {te_metric:.4f}")


def tune_source(dataset: Dataset, config: ModelConfig):
    if config.src_tune_config is None:
        raise ValueError()

    # wandb login
    if not dotenv.load_dotenv():
        logger.error("No environment variables found")
        exit(1)
    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        logger.error("Missing Wandb API key in the environment file")
        exit(1)
    wandb.login(key=api_key)

    mf_tuning(
        seed=config.seed,
        tune_config=config.get_wandb_dict("source"),
        train_set=dataset.src_tr,
        val_set=dataset.src_val,
        val_batch_size=config.src_train_config.batch_size,
        n_users=dataset.src_n_users,
        n_items=dataset.src_n_items,
        ui_matrix=dataset.src_ui_matrix,
        metric=config.val_metric,
        n_epochs=config.epochs,
        early=config.early_stopping_patience,
        early_stopping_criterion=config.early_stopping_criterion,
        entity_name=config.src_tune_config.entity_name,
        exp_name=config.src_tune_config.exp_name,
        bayesian_run_count=config.src_tune_config.bayesian_run_count,
        sweep_id=config.src_tune_config.sweep_id,
    )


if __name__ == "__main__":
    main_source()
