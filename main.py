import argparse
from pathlib import Path

import orjson
import torch
import wandb
from loguru import logger

from src.ModelConfig import ModelConfig
from src.bpr_loss import BPRLoss
from src.data_preprocessing import SourceTargetDatasets, process_source_target
from src.loader import DataLoader
from src.models.mf import MFTrainer, MatrixFactorization
from src.tuning import mf_tuning
from src.utils import set_seed
import dotenv
import os

parser = argparse.ArgumentParser(description="PyTorch BPR Training")
parser.add_argument("--config", help="path to config file", required=True)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
parser.add_argument("--clear", help="clear saved data", action="store_true")


def main():
    args = parser.parse_args()
    config_file_path = args.config
    with open(config_file_path, "rb") as config_file:
        config_json = orjson.loads(config_file.read())
        config = ModelConfig(config_json)

    dataset = process_source_target(
        seed=config.seed,
        source_dataset_path=config.src_ratings_path,
        target_dataset_path=config.tgt_ratings_path,
        paths_file_path=config.paths_file_path,
        save_path=Path("./data/saved_data/"),
        clear_saved_dataset=args.clear
    )

    if args.train:
        train_source(dataset, config)
    elif args.tune:
        tune_source(dataset, config)


def train_source(dataset: SourceTargetDatasets, config: ModelConfig):
    logger.info(f"Training the model with configuration: {config.get_train_config()}")
    set_seed(config.seed)

    tr_loader = DataLoader(
        data=dataset["src_tr"],
        ui_matrix=dataset["src_ui_matrix"],
        batch_size=config.batch_size,
    )

    val_loader = DataLoader(
        data=dataset["src_val"],
        ui_matrix=dataset["src_ui_matrix"],
        batch_size=config.batch_size,
    )

    te_loader = DataLoader(
        data=dataset["src_te"],
        ui_matrix=dataset["src_ui_matrix"],
        batch_size=config.batch_size
    )

    mf = MatrixFactorization(
        n_users=dataset["src_n_users"],
        n_items=dataset["src_n_items"],
        n_factors=config.n_factors,
    )

    tr = MFTrainer(
        mf_model=mf,
        optimizer=torch.optim.AdamW(
            mf.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        ),
        loss=BPRLoss(),
    )

    tr.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        early=config.early_stopping_patience,
        verbose=1,
        early_loss_based=config.early_stopping_loss,
        save_path=config.save_path
    )

    te_metric, _ = tr.validate(te_loader, val_metric=config.val_metric)

    print("Test %s: %.4f" % (config.val_metric, te_metric))


def tune_source(dataset: SourceTargetDatasets, config: ModelConfig):
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
        tune_config=config.sweep_config,
        train_set=dataset["src_tr"],
        val_set=dataset["src_val"],
        val_batch_size=config.batch_size,
        n_users=dataset["src_n_users"],
        n_items=dataset["src_n_items"],
        ui_matrix=dataset["src_ui_matrix"],
        metric=config.val_metric,
        n_epochs=config.epochs,
        early=config.early_stopping_patience,
        early_loss_based=config.early_stopping_loss,
        entity_name=config.entity_name,
        exp_name=config.exp_name,
        bayesian_run_count=config.bayesian_run_count,
        sweep_id=config.sweep_id
    )


if __name__ == "__main__":
    main()
