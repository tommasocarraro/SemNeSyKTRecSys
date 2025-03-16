import os
from typing import Optional

import dotenv
import torch
import wandb
from loguru import logger

from src.data_loader import TrDataLoader, ValDataLoader
from src.data_preprocessing.Dataset import DatasetComparison
from src.evaluation import evaluate_model
from src.model import MatrixFactorization
from src.model_configs.mf.ModelConfigMf import ModelConfigMf
from src.pretrain_source.loss import BPRLoss
from src.pretrain_source.mf_trainer import MfTrainer
from src.pretrain_source.tuning import mf_tuning


def _get_trainer_loaders(dataset: DatasetComparison, config: ModelConfigMf):
    """
    Creates the data loaders and the trainer for the MF model

    :param dataset: Dataset object
    :param config: Model configuration
    :return: Trainer, train loader, validation loader, test loader, test loader with shared users ratings only
    """
    tr = dataset.tr_no_sh
    val = dataset.val_no_sh
    te = dataset.te
    te_sh = dataset.te_only_sh

    hyperparams = config.train_config.hyper_params

    tr_loader = TrDataLoader(data=tr, ui_matrix=dataset.ui_matrix_no_sh, batch_size=hyperparams.batch_size)
    val_loader = ValDataLoader(data=val, ui_matrix=dataset.ui_matrix_no_sh, batch_size=hyperparams.batch_size)
    te_loader = ValDataLoader(data=te, ui_matrix=dataset.ui_matrix, batch_size=hyperparams.batch_size)
    te_loader_sh = ValDataLoader(data=te_sh, ui_matrix=dataset.ui_matrix, batch_size=hyperparams.batch_size)

    mf = MatrixFactorization(n_users=dataset.n_users, n_items=dataset.n_items, n_factors=hyperparams.n_factors)

    trainer = MfTrainer(
        model=mf,
        optimizer=torch.optim.AdamW(
            mf.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
        ),
        loss=BPRLoss(),
    )

    return trainer, tr_loader, val_loader, te_loader, te_loader_sh


def train_mf(dataset: DatasetComparison, config: ModelConfigMf):
    """
    Trains the MF model

    :param dataset: Dataset object
    :param config: Model configuration
    """
    logger.info(f"Training the target MF model with configuration: {config.get_train_config_str()}")

    trainer, tr_loader, val_loader, te_loader, te_loader_sh = _get_trainer_loaders(dataset=dataset, config=config)

    trainer.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        checkpoint_save_path=config.train_config.checkpoint_save_path,
        final_model_save_path=config.train_config.final_model_save_path,
    )

    val_metric_results, _ = trainer.validate(val_loader=val_loader, val_metric=config.val_metric, use_val_loss=False)

    logger.info(f"Training complete. Final validation {config.val_metric.value}: {val_metric_results:.5f}")

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        val_metric=config.val_metric,
        log_string=f"Evaluating BPR-MF on {dataset.train_dataset_name} dataset, shared users with "
        f"{dataset.other_dataset_name} dataset and {dataset.sparsity_sh * 100}% of shared users ratings",
    )


def tune_mf(dataset: DatasetComparison, config: ModelConfigMf, sweep_id: Optional[str], sweep_name: Optional[str]):
    """
    Tunes the MF model

    :param dataset: Dataset object
    :param config: Model configuration
    :param sweep_id: Sweep ID
    :param sweep_name: Sweep name
    """
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

    mf_tuning(
        tune_config=config.get_wandb_dict_mf(),
        train_set=dataset.tr_no_sh,
        val_set=dataset.val_no_sh,
        val_batch_size=config.train_config.hyper_params.batch_size,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        ui_matrix=dataset.ui_matrix_no_sh,
        metric=config.val_metric,
        n_epochs=config.epochs,
        entity_name=config.mf_tune_config.entity_name,
        exp_name=config.mf_tune_config.exp_name,
        bayesian_run_count=config.mf_tune_config.bayesian_run_count,
        sweep_id=sweep_id or config.mf_tune_config.sweep_id,
        sweep_name=sweep_name,
    )


def test_mf(dataset: DatasetComparison, config: ModelConfigMf):
    """
    Tests the MF model

    :param dataset: Dataset object
    :param config: Model configuration
    """
    trainer, _, _, te_loader, te_loader_sh = _get_trainer_loaders(dataset=dataset, config=config)

    return evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        weights_path=config.train_config.final_model_save_path,
        val_metric=config.val_metric,
        log_string=f"Evaluating BPR-MF on {dataset.train_dataset_name} dataset, shared users with "
        f"{dataset.other_dataset_name} dataset and {dataset.sparsity_sh * 100}% of shared users ratings",
    )
