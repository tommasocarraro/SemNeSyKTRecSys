import argparse
import os
from pathlib import Path
from typing import Literal

import dotenv
import torch
import wandb
from loguru import logger

from data_preprocessing import Dataset, process_source
from source_pretrain.data_loader import DataLoader
from source_pretrain.loss import BPRLoss
from source_pretrain.model import MatrixFactorization
from source_pretrain.model_configs import ModelConfig, get_config
from source_pretrain.trainer import MfTrainer
from source_pretrain.tuning import mf_tuning
from utils import set_seed

parser = argparse.ArgumentParser(description="PyTorch BPR Training")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
parser.add_argument(
    "dataset",
    type=str,
    help="Dataset to use (appears after --train or --tune)",
    nargs="?",
    choices=["movies", "music", "books"],
)
parser.add_argument("--clear", help="recompute dataset", action="store_true")


def main_source():
    args = parser.parse_args()

    dataset_name = args.dataset
    kind: Literal["train", "tune"] = "train" if args.train else "tune"

    config = get_config(dataset_name=dataset_name, kind=kind)

    dataset = process_source(
        seed=config.seed,
        dataset_path=config.dataset_path,
        save_path=Path("../data/saved_data/"),
        clear_saved_dataset=args.clear,
    )

    if args.train:
        train_source(dataset, config)
    elif args.tune:
        tune_source(dataset, config)


def train_source(dataset: Dataset, config: ModelConfig):
    logger.info(f"Training the model with configuration: {config.get_train_config()}")
    set_seed(config.seed)

    tr_loader = DataLoader(
        data=dataset.tr,
        ui_matrix=dataset.ui_matrix,
        batch_size=config.train_config.batch_size,
    )

    val_loader = DataLoader(
        data=dataset.val,
        ui_matrix=dataset.ui_matrix,
        batch_size=config.train_config.batch_size,
    )

    mf = MatrixFactorization(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_factors=config.train_config.n_factors,
    )

    tr = MfTrainer(
        model=mf,
        optimizer=torch.optim.AdamW(
            mf.parameters(),
            lr=config.train_config.learning_rate,
            weight_decay=config.train_config.weight_decay,
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
        save_path=config.train_config.model_save_path,
    )

    if dataset.te is not None:
        te_loader = DataLoader(
            data=dataset.te,
            ui_matrix=dataset.ui_matrix,
            batch_size=config.train_config.batch_size,
        )
        te_metric, _ = tr.validate(te_loader, val_metric=config.val_metric)
        logger.info(f"Test {config.val_metric}: {te_metric:.4f}")


def tune_source(dataset: Dataset, config: ModelConfig):
    if config.tune_config is None:
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
        tune_config=config.get_wandb_dict(),
        train_set=dataset.tr,
        val_set=dataset.val,
        val_batch_size=config.train_config.batch_size,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        ui_matrix=dataset.ui_matrix,
        metric=config.val_metric,
        n_epochs=config.epochs,
        early=config.early_stopping_patience,
        early_stopping_criterion=config.early_stopping_criterion,
        entity_name=config.tune_config.entity_name,
        exp_name=config.tune_config.exp_name,
        bayesian_run_count=config.tune_config.bayesian_run_count,
        sweep_id=config.tune_config.sweep_id,
    )


if __name__ == "__main__":
    main_source()
