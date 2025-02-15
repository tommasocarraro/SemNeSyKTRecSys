import os
from typing import Literal, Optional

import dotenv
import torch
import wandb
from loguru import logger

from src.data_loader import TrDataLoader, ValDataLoader
from src.data_preprocessing.Dataset import Dataset
from src.model import MatrixFactorization
from src.model_configs import ModelConfig
from src.pretrain_source.loss import BPRLoss
from src.pretrain_source.mf_trainer import MfTrainer
from src.pretrain_source.tuning import mf_tuning


def _create_trainer(dataset: Dataset, config: ModelConfig, which_dataset: Literal["source", "target"]):
    if which_dataset == "source":
        tr = dataset.src_tr
        val = dataset.src_val
        te = dataset.src_te
        ui_matrix = dataset.src_ui_matrix
        n_users = dataset.src_n_users
        n_items = dataset.src_n_items
    else:
        tr = dataset.tgt_tr
        val = dataset.tgt_val
        te = dataset.tgt_te
        ui_matrix = dataset.tgt_ui_matrix
        n_users = dataset.tgt_n_users
        n_items = dataset.tgt_n_items

    hyperparams = config.mf_train_config.hyper_params

    tr_loader = TrDataLoader(data=tr, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)
    val_loader = ValDataLoader(data=val, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)
    te_loader = ValDataLoader(data=te, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)

    mf = MatrixFactorization(n_users=n_users, n_items=n_items, n_factors=hyperparams.n_factors)

    tr = MfTrainer(
        model=mf,
        optimizer=torch.optim.AdamW(
            mf.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
        ),
        loss=BPRLoss(),
    )

    return tr, tr_loader, val_loader, te_loader


def train_mf(dataset: Dataset, config: ModelConfig, which_dataset: Literal["source", "target"]):
    logger.info(f"Training the model with configuration: {config.get_train_config_str('mf')}")

    tr, tr_loader, val_loader, te_loader = _create_trainer(dataset=dataset, config=config, which_dataset=which_dataset)

    tr.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        early=config.early_stopping_patience,
        verbose=1,
        early_stopping_criterion=config.early_stopping_criterion,
        checkpoint_save_path=config.mf_train_config.checkpoint_save_path,
        final_model_save_path=config.mf_train_config.final_model_save_path,
    )

    val_metric_results, _ = tr.validate(val_loader=val_loader, val_metric=config.val_metric, use_val_loss=False)

    logger.info(f"Training complete. Final validation {config.val_metric.name}: {val_metric_results:.4f}")

    te_metric_results, _ = tr.validate(te_loader, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric.name}: {te_metric_results:.4f}")


def tune_mf(
    dataset: Dataset,
    config: ModelConfig,
    which_dataset: Literal["source", "target"],
    sweep_id: Optional[str],
    sweep_name: Optional[str],
):
    if config.mf_tune_config is None:
        raise ValueError("Missing tuning configuration")

    # wandb login
    if not dotenv.load_dotenv():
        logger.error("No environment variables found")
        exit(1)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        logger.error("Missing Wandb API key in the environment file")
        exit(1)
    wandb.login(key=wandb_api_key)

    if which_dataset == "source":
        tr = dataset.src_tr
        val = dataset.src_val
        ui_matrix = dataset.src_ui_matrix
        n_users = dataset.src_n_users
        n_items = dataset.src_n_items
    else:
        tr = dataset.tgt_tr
        val = dataset.tgt_val
        ui_matrix = dataset.tgt_ui_matrix
        n_users = dataset.tgt_n_users
        n_items = dataset.tgt_n_items

    mf_tuning(
        tune_config=config.get_wandb_dict_mf(),
        train_set=tr,
        val_set=val,
        val_batch_size=config.mf_train_config.hyper_params.batch_size,
        n_users=n_users,
        n_items=n_items,
        ui_matrix=ui_matrix,
        metric=config.val_metric,
        n_epochs=config.epochs,
        early=config.early_stopping_patience,
        early_stopping_criterion=config.early_stopping_criterion,
        entity_name=config.mf_tune_config.entity_name,
        exp_name=config.mf_tune_config.exp_name,
        bayesian_run_count=config.mf_tune_config.bayesian_run_count,
        sweep_id=sweep_id or config.mf_tune_config.sweep_id,
        sweep_name=sweep_name,
    )


def test_mf(dataset: Dataset, config: ModelConfig, which_dataset: Literal["source", "target"]):
    tr, _, _, te_loader = _create_trainer(dataset=dataset, config=config, which_dataset=which_dataset)

    weights_path = config.mf_train_config.final_model_save_path
    tr.model.load_model(weights_path)

    te_metric_results, _ = tr.validate(te_loader, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric.name}: {te_metric_results:.4f}")
