import os
from typing import Optional

import dotenv
import torch
import wandb
from loguru import logger
from torch import Tensor
from torch.optim import AdamW

from src.cross_domain.ltn_trainer import LTNRegTrainer
from src.cross_domain.tuning import ltn_tuning_reg
from src.data_loader import TrDataLoader, ValDataLoader
from src.data_preprocessing.Dataset import DatasetPretrain, DatasetTarget
from src.evaluation import evaluate_model
from src.model import MatrixFactorization
from src.model_configs.ltn.ModelConfigLtn import ModelConfigLtn
from src.model_configs.mf import ModelConfigMf
from src.pretrain_source.loss import BPRLoss
from src.pretrain_source.mf_trainer import MfTrainer


def _get_trainer_loaders(dataset: DatasetTarget, config: ModelConfigLtn, processed_interactions: dict[int, Tensor]):
    """
    Creates the data loaders and the trainer for the LTN model

    :param dataset: Dataset object
    :param config: Model configuration
    :param processed_interactions: Processed interactions from source model
    :return: Trainer, train loader, validation loader, test loader, test loader with shared users ratings only
    """
    hyperparams = config.tgt_train_config.hyper_params

    tr_loader = TrDataLoader(
        data=dataset.tgt_tr_no_sh,
        ui_matrix=dataset.tgt_ui_matrix_no_sh,
        batch_size=hyperparams.batch_size,
        processed_interactions=processed_interactions,
    )

    val_loader = ValDataLoader(
        data=dataset.tgt_val_no_sh, ui_matrix=dataset.tgt_ui_matrix_no_sh, batch_size=hyperparams.batch_size
    )

    te_loader = ValDataLoader(data=dataset.tgt_te, ui_matrix=dataset.tgt_ui_matrix, batch_size=hyperparams.batch_size)

    te_loader_sh = ValDataLoader(
        data=dataset.tgt_te_only_sh, ui_matrix=dataset.tgt_ui_matrix, batch_size=hyperparams.batch_size
    )

    mf_model_tgt = MatrixFactorization(
        n_users=dataset.n_users, n_items=dataset.n_items, n_factors=hyperparams.n_factors
    )

    trainer = LTNRegTrainer(
        mf_model=mf_model_tgt,
        optimizer=torch.optim.AdamW(
            mf_model_tgt.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
        ),
        p_forall_ax1=hyperparams.p_forall_ax1,
        p_forall_ax2=hyperparams.p_forall_ax2,
        p_sat_agg=hyperparams.p_sat_agg,
        tgt_ui_matrix=dataset.tgt_ui_matrix_no_sh,
    )

    return trainer, tr_loader, val_loader, te_loader, te_loader_sh


def train_ltn(dataset: DatasetTarget, config: ModelConfigLtn, processed_interactions: dict[int, Tensor]):
    """
    Trains the LTN model

    :param dataset: Dataset object
    :param config: Model configuration
    :param processed_interactions: Processed interactions from source model
    """
    train_config = config.tgt_train_config
    logger.info(f"Training the LTN model with configuration: {config.get_train_config_str()}")

    trainer, tr_loader, val_loader, te_loader, te_loader_sh = _get_trainer_loaders(
        dataset=dataset, config=config, processed_interactions=processed_interactions
    )

    trainer.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric=config.val_metric,
        checkpoint_save_path=train_config.checkpoint_save_path,
        final_model_save_path=train_config.final_model_save_path,
    )

    val_metric_results, _ = trainer.validate(val_loader=val_loader, val_metric=config.val_metric)
    logger.info(f"Training complete. Final validation {config.val_metric.value}: {val_metric_results:.5f}")

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        val_metric=config.val_metric,
        log_string=f"Evaluating LTN REG model on {dataset.src_dataset_name}->{dataset.tgt_dataset_name} datasets, shared"
        f" users with {dataset.tgt_dataset_name} dataset and {dataset.sparsity_sh*100}% of shared users ratings",
    )


def tune_ltn(
    dataset: DatasetTarget,
    config: ModelConfigLtn,
    processed_interactions: dict[int, Tensor],
    sweep_id: Optional[str],
    sweep_name: Optional[str],
):
    """
    Tunes the LTN model

    :param dataset: Dataset object
    :param config: Model configuration
    :param processed_interactions: Processed interactions from source model
    :param sweep_id: Sweep ID
    :param sweep_name: Sweep name
    """
    train_config = config.tgt_train_config
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
        tune_config=config.get_wandb_dict_ltn(),
        train_set=dataset.tgt_tr_no_sh,
        val_set=dataset.tgt_val_no_sh,
        val_batch_size=train_config.hyper_params.batch_size,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        tgt_ui_matrix=dataset.tgt_ui_matrix_no_sh,
        val_metric=config.val_metric,
        n_epochs=config.epochs,
        entity_name=tune_config.entity_name,
        exp_name=tune_config.exp_name,
        bayesian_run_count=tune_config.bayesian_run_count,
        sweep_id=sweep_id or tune_config.sweep_id,
        sweep_name=sweep_name,
        processed_interactions=processed_interactions,
    )


def test_ltn(dataset: DatasetTarget, config: ModelConfigLtn, processed_interactions: dict[int, Tensor]):
    """
    Tests the LTN model

    :param dataset: Dataset object
    :param config: Model configuration
    :param processed_interactions: Processed interactions from source model
    """
    trainer, _, _, te_loader, te_loader_sh = _get_trainer_loaders(
        dataset=dataset, config=config, processed_interactions=processed_interactions
    )

    evaluate_model(
        trainer=trainer,
        te_loader=te_loader,
        te_loader_sh=te_loader_sh,
        weights_path=config.tgt_train_config.final_model_save_path,
        val_metric=config.val_metric,
        log_string=f"Evaluating LTN REG model on {dataset.src_dataset_name}->{dataset.tgt_dataset_name} datasets, shared"
        f"users with {dataset.tgt_dataset_name} dataset and {dataset.sparsity_sh * 100}% of shared users ratings",
    )


def pretrain_mf_for_ltn(dataset: DatasetPretrain, config: ModelConfigMf) -> MatrixFactorization:
    """
    Trains the MF model used by LTN to generate the processed interactions

    :param dataset: Dataset object
    :param config: Model configuration
    :return: The trained MF model
    """
    mf_pretrain = MatrixFactorization(
        n_users=dataset.n_users, n_items=dataset.n_items, n_factors=config.train_config.hyper_params.n_factors
    )
    if config.train_config.final_model_save_path.is_file():
        mf_pretrain.load_model(config.train_config.final_model_save_path)
        return mf_pretrain
    print()
    logger.warning(f"Source MF model for LTN not found. Training it now...")
    logger.info(f"Training the source MF model with configuration: {config.get_train_config_str()}")

    trainer_mf_pretrain = MfTrainer(
        model=mf_pretrain,
        optimizer=AdamW(
            params=mf_pretrain.parameters(),
            lr=config.train_config.hyper_params.learning_rate,
            weight_decay=config.train_config.hyper_params.weight_decay,
        ),
        loss=BPRLoss(),
    )
    pretrain_tr_loader = TrDataLoader(
        data=dataset.tr, ui_matrix=dataset.ui_matrix, batch_size=config.train_config.hyper_params.batch_size
    )
    pretrain_val_loader = ValDataLoader(
        data=dataset.val, ui_matrix=dataset.ui_matrix, batch_size=config.train_config.hyper_params.batch_size
    )
    trainer_mf_pretrain.train(
        train_loader=pretrain_tr_loader,
        val_loader=pretrain_val_loader,
        val_metric=config.val_metric,
        checkpoint_save_path=config.train_config.checkpoint_save_path,
        final_model_save_path=config.train_config.final_model_save_path,
    )
    logger.info("Training of source MF model complete")
    print()
    return mf_pretrain
