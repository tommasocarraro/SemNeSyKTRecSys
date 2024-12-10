import argparse
from pathlib import Path

import orjson
import torch
from loguru import logger

from src.ModelConfig import ModelConfig
from src.bpr_loss import BPRLoss
from src.data_preprocessing import SourceTargetDatasets, process_source_target
from src.loader import DataLoader
from src.models.mf import MFTrainer, MatrixFactorization
from src.tuning import mf_tuning
from src.utils import set_seed

parser = argparse.ArgumentParser(description="PyTorch BPR Training")
parser.add_argument("--config", help="path to config file", required=True)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")


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
    )

    if args.train:
        train_source(dataset, config)
    elif args.tune:
        tune_source(dataset, config)


def train_source(dataset: SourceTargetDatasets, config: ModelConfig):
    logger.info("Training the model...")
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
    )


def tune_source(dataset: SourceTargetDatasets, config: ModelConfig):
    mf_tuning(
        seed=config.seed,
        tune_config=config.sweep_config,
        train_set=dataset["src_tr"],
        val_set=dataset["src_val"],
        n_users=dataset["src_n_users"],
        n_items=dataset["src_n_items"],
        ui_matrix=dataset["src_ui_matrix"],
        metric=config.val_metric,
        entity_name=config.entity_name,
        exp_name=config.exp_name,
    )


if __name__ == "__main__":
    main()
