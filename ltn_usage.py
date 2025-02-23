import os
from typing import Optional

import dotenv
import torch
import wandb
from loguru import logger
from torch import Tensor

from src.cross_domain.ltn_trainer import LTNRegTrainer
from src.cross_domain.tuning import ltn_tuning_reg
from src.data_loader import TrDataLoader, ValDataLoader
from src.data_preprocessing.Dataset import DatasetLtn
from src.evaluation import evaluate_model
from src.model import MatrixFactorization
from src.model_configs.ltn.ModelConfigLtn import ModelConfigLtn


def _get_trainer_loaders(dataset: DatasetLtn, config: ModelConfigLtn, processed_interactions: dict[int, Tensor]):
    hyperparams = config.train_config.hyper_params_target

    tr_loader = TrDataLoader(
        data=dataset.tgt_tr,
        ui_matrix=dataset.tgt_ui_matrix,
        batch_size=hyperparams.batch_size,
        processed_interactions=processed_interactions,
    )

    val_loader = ValDataLoader(data=dataset.tgt_val, ui_matrix=dataset.tgt_ui_matrix, batch_size=hyperparams.batch_size)

    te_loader = ValDataLoader(data=dataset.tgt_te, ui_matrix=dataset.tgt_ui_matrix, batch_size=hyperparams.batch_size)

    te_loader_sh = ValDataLoader(
        data=dataset.tgt_te_sh, ui_matrix=dataset.tgt_ui_matrix, batch_size=hyperparams.batch_size
    )

    mf_model_tgt = MatrixFactorization(
        n_users=dataset.tgt_n_users, n_items=dataset.tgt_n_items, n_factors=hyperparams.n_factors
    )

    trainer = LTNRegTrainer(
        mf_model=mf_model_tgt,
        optimizer=torch.optim.AdamW(
            mf_model_tgt.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
        ),
        p_forall_ax1=hyperparams.p_forall_ax1,
        p_forall_ax2=hyperparams.p_forall_ax2,
        p_sat_agg=hyperparams.p_sat_agg,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
    )

    return trainer, tr_loader, val_loader, te_loader, te_loader_sh


def train_ltn_reg(dataset: DatasetLtn, config: ModelConfigLtn, processed_interactions: dict[int, Tensor]):
    train_config = config.train_config
    logger.info(f"Training the model with configuration: {config.get_train_config_str()}")

    trainer, tr_loader, val_loader, te_loader, te_loader_sh = _get_trainer_loaders(
        dataset=dataset, config=config, processed_interactions=processed_interactions
    )

    trainer.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        early=config.early_stopping_patience,
        verbose=1,
        early_stopping_criterion=config.early_stopping_criterion,
        checkpoint_save_path=train_config.checkpoint_save_path,
        final_model_save_path=train_config.final_model_save_path,
    )

    val_metric_results, _ = trainer.validate(val_loader=val_loader, val_metric=config.val_metric, use_val_loss=False)
    logger.info(f"Training complete. Final validation {config.val_metric.name}: {val_metric_results:.5f}")

    te_metric_results, _ = trainer.validate(te_loader, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric.name}: {te_metric_results:.5f}")

    te_sh_metric_results, _ = trainer.validate(te_loader_sh, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric.name} on shared users only: {te_sh_metric_results:.5f}")

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        val_metric=config.val_metric,
        model_name="LTN REG",
    )


def tune_ltn_reg(
    dataset: DatasetLtn,
    config: ModelConfigLtn,
    processed_interactions: dict[int, Tensor],
    sweep_id: Optional[str],
    sweep_name: Optional[str],
):
    train_config = config.train_config
    tune_config = config.ltn_reg_tune_config
    if tune_config is None:
        logger.error("Missing tuning configuration")
        exit(1)

    # wandb login
    if not dotenv.load_dotenv():
        logger.error("No environment variables found")
        exit(1)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        logger.error("Missing Wandb API key in the environment file")
        exit(1)
    wandb.login(key=wandb_api_key)

    ltn_tuning_reg(
        tune_config=config.get_wandb_dict_ltn_reg(),
        train_set=dataset.tgt_tr,
        val_set=dataset.tgt_val,
        val_batch_size=train_config.hyper_params_target.batch_size,
        n_users=dataset.tgt_n_users,
        n_items=dataset.tgt_n_items,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
        val_metric=config.val_metric,
        n_epochs=config.epochs,
        early=config.early_stopping_patience,
        early_stopping_criterion=config.early_stopping_criterion,
        entity_name=tune_config.entity_name,
        exp_name=tune_config.exp_name,
        bayesian_run_count=tune_config.bayesian_run_count,
        sweep_id=sweep_id or tune_config.sweep_id,
        sweep_name=sweep_name,
        processed_interactions=processed_interactions,
    )


def test_ltn_reg(dataset: DatasetLtn, config: ModelConfigLtn, processed_interactions: dict[int, Tensor]):
    trainer, _, _, te_loader, te_loader_sh = _get_trainer_loaders(
        dataset=dataset, config=config, processed_interactions=processed_interactions
    )

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        weights_path=config.train_config.final_model_save_path,
        val_metric=config.val_metric,
        model_name="LTN Reg",
    )
