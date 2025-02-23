import os
from typing import Optional

import dotenv
import torch
import wandb
from loguru import logger

from src.data_loader import TrDataLoader, ValDataLoader
from src.data_preprocessing.Dataset import DatasetMf
from src.evaluation import evaluate_model
from src.model import MatrixFactorization
from src.model_configs.mf.ModelConfigMf import ModelConfigMf
from src.pretrain_source.loss import BPRLoss
from src.pretrain_source.mf_trainer import MfTrainer
from src.pretrain_source.tuning import mf_tuning


def _get_trainer_loaders(dataset: DatasetMf, config: ModelConfigMf):
    tr = dataset.tr
    val = dataset.val
    te = dataset.te
    te_sh = dataset.te_sh
    ui_matrix = dataset.ui_matrix
    n_users = dataset.n_users
    n_items = dataset.n_items

    hyperparams = config.train_config.hyper_params

    tr_loader = TrDataLoader(data=tr, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)
    val_loader = ValDataLoader(data=val, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)
    te_loader = ValDataLoader(data=te, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)
    te_loader_sh = ValDataLoader(data=te_sh, ui_matrix=ui_matrix, batch_size=hyperparams.batch_size)

    mf = MatrixFactorization(n_users=n_users, n_items=n_items, n_factors=hyperparams.n_factors)

    trainer = MfTrainer(
        model=mf,
        optimizer=torch.optim.AdamW(
            mf.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
        ),
        loss=BPRLoss(),
    )

    return trainer, tr_loader, val_loader, te_loader, te_loader_sh


def train_mf(dataset: DatasetMf, config: ModelConfigMf):
    logger.info(f"Training the model with configuration: {config.get_train_config_str()}")

    trainer, tr_loader, val_loader, te_loader, te_loader_sh = _get_trainer_loaders(dataset=dataset, config=config)

    trainer.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        early=config.early_stopping_patience,
        verbose=1,
        early_stopping_criterion=config.early_stopping_criterion,
        checkpoint_save_path=config.train_config.checkpoint_save_path,
        final_model_save_path=config.train_config.final_model_save_path,
    )

    val_metric_results, _ = trainer.validate(val_loader=val_loader, val_metric=config.val_metric, use_val_loss=False)

    logger.info(f"Training complete. Final validation {config.val_metric.name}: {val_metric_results:.5f}")

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        val_metric=config.val_metric,
        model_name="BPR-MF",
    )


def tune_mf(dataset: DatasetMf, config: ModelConfigMf, sweep_id: Optional[str], sweep_name: Optional[str]):
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

    tr = dataset.tr
    val = dataset.val
    ui_matrix = dataset.ui_matrix
    n_users = dataset.n_users
    n_items = dataset.n_items

    mf_tuning(
        tune_config=config.get_wandb_dict_mf(),
        train_set=tr,
        val_set=val,
        val_batch_size=config.train_config.hyper_params.batch_size,
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


def test_mf(dataset: DatasetMf, config: ModelConfigMf):
    trainer, _, _, te_loader, te_loader_sh = _get_trainer_loaders(dataset=dataset, config=config)

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        weights_path=config.train_config.final_model_save_path,
        val_metric=config.val_metric,
        model_name="BPR-MF",
    )
