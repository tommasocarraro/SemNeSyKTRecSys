import os
from pathlib import Path
from typing import Optional

import dotenv
import torch
import wandb
from loguru import logger

from src.cross_domain.ltn_trainer import LTNRegTrainer
from src.cross_domain.tuning import ltn_tuning_reg
from src.cross_domain.utils import get_reg_axiom_data
from src.data_loader import DataLoader, ValDataLoader
from src.data_preprocessing.Dataset import Dataset
from src.model import MatrixFactorization
from src.model_configs import ModelConfig
from src.model_configs.utils import Domains_Type, get_best_weights_path_mf_source
from src.pretrain_source.inference import generate_pre_trained_src_matrix


def _create_trainer(
    dataset: Dataset,
    config: ModelConfig,
    src_dataset_name: Domains_Type,
    tgt_dataset_name: Domains_Type,
    src_sparsity: float,
    user_level_src: bool,
    tgt_sparsity: float,
    save_dir_path: Path,
):
    tr_loader = DataLoader(
        data=dataset.tgt_tr,
        ui_matrix=dataset.tgt_ui_matrix,
        batch_size=config.ltn_reg_train_config.hyper_params.batch_size,
    )

    val_loader = ValDataLoader(
        data=dataset.tgt_val,
        ui_matrix=dataset.tgt_ui_matrix,
        batch_size=config.ltn_reg_train_config.hyper_params.batch_size,
    )

    mf_model_tgt = MatrixFactorization(
        n_users=dataset.tgt_n_users,
        n_items=dataset.tgt_n_items,
        n_factors=config.ltn_reg_train_config.hyper_params.n_factors,
    )

    mf_model_src = MatrixFactorization(
        n_users=dataset.src_n_users,
        n_items=dataset.src_n_items,
        n_factors=config.mf_train_config.hyper_params.n_factors,
    )

    src_model_weights_path = get_best_weights_path_mf_source(
        src_domain_name=src_dataset_name, src_sparsity=src_sparsity, user_level_src=user_level_src
    )
    if not src_model_weights_path or not src_model_weights_path.is_file():
        logger.error(f"The source model weights path {src_model_weights_path} does not exist.")
        exit(1)

    top_k_items = generate_pre_trained_src_matrix(
        mf_model=mf_model_src,
        best_weights_path=src_model_weights_path,
        n_shared_users=dataset.n_sh_users,
        save_dir_path=save_dir_path,
        batch_size=2048,
        k=config.ltn_reg_train_config.hyper_params.top_k_src,
    )

    processed_interactions = get_reg_axiom_data(
        src_ui_matrix=dataset.src_ui_matrix,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
        n_sh_users=dataset.n_sh_users,
        sim_matrix=dataset.sim_matrix,
        top_k_items=top_k_items,
        save_dir_path=save_dir_path,
        src_dataset_name=src_dataset_name,
        tgt_dataset_name=tgt_dataset_name,
        tgt_sparsity=tgt_sparsity,
    )

    tr = LTNRegTrainer(
        mf_model=mf_model_tgt,
        optimizer=torch.optim.AdamW(
            mf_model_tgt.parameters(),
            lr=config.ltn_reg_train_config.hyper_params.learning_rate,
            weight_decay=config.ltn_reg_train_config.hyper_params.weight_decay,
        ),
        p_forall_ax1=config.ltn_reg_train_config.hyper_params.p_forall_ax1,
        p_forall_ax2=config.ltn_reg_train_config.hyper_params.p_forall_ax2,
        p_sat_agg=config.ltn_reg_train_config.hyper_params.p_sat_agg,
        neg_score_value=config.ltn_reg_train_config.hyper_params.neg_score,
        processed_interactions=processed_interactions,
    )

    return tr, tr_loader, val_loader


def train_ltn_reg(
    dataset: Dataset,
    config: ModelConfig,
    src_dataset_name: Domains_Type,
    tgt_dataset_name: Domains_Type,
    src_sparsity: float,
    user_level_src: bool,
    tgt_sparsity: float,
    save_dir_path: Path,
):
    train_config = config.ltn_reg_train_config
    logger.info(f"Training the model with configuration: {config.get_train_config_str('ltn_reg')}")

    tr, tr_loader, val_loader = _create_trainer(
        dataset=dataset,
        config=config,
        src_dataset_name=src_dataset_name,
        tgt_dataset_name=tgt_dataset_name,
        src_sparsity=src_sparsity,
        tgt_sparsity=tgt_sparsity,
        save_dir_path=save_dir_path,
        user_level_src=user_level_src,
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
        data=dataset.tgt_te, ui_matrix=dataset.tgt_ui_matrix, batch_size=train_config.hyper_params.batch_size
    )
    te_metric_results, _ = tr.validate(te_loader, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric.name}: {te_metric_results:.4f}")


def tune_ltn_reg(
    dataset: Dataset,
    config: ModelConfig,
    src_dataset_name: Domains_Type,
    tgt_dataset_name: Domains_Type,
    src_sparsity: float,
    user_level_src: bool,
    tgt_sparsity: float,
    sweep_id: Optional[str],
    sweep_name: Optional[str],
    save_dir_path: Path,
):
    train_config = config.ltn_reg_train_config
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

    mf_model_src = MatrixFactorization(
        n_users=dataset.src_n_users,
        n_items=dataset.src_n_items,
        n_factors=config.mf_train_config.hyper_params.n_factors,
    )

    src_model_weights_path = get_best_weights_path_mf_source(
        src_domain_name=src_dataset_name, src_sparsity=src_sparsity, user_level_src=user_level_src
    )
    if not src_model_weights_path or not src_model_weights_path.is_file():
        logger.error(f"The source model weights path {src_model_weights_path} does not exist.")
        exit(1)

    ltn_tuning_reg(
        tune_config=config.get_wandb_dict_ltn_reg(),
        train_set=dataset.tgt_tr,
        val_set=dataset.tgt_val,
        val_batch_size=train_config.hyper_params.batch_size,
        n_users=dataset.tgt_n_users,
        n_items=dataset.tgt_n_items,
        src_ui_matrix=dataset.src_ui_matrix,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
        val_metric=config.val_metric,
        n_epochs=config.epochs,
        early=config.early_stopping_patience,
        early_stopping_criterion=config.early_stopping_criterion,
        entity_name=tune_config.entity_name,
        exp_name=tune_config.exp_name,
        bayesian_run_count=tune_config.bayesian_run_count,
        sweep_id=sweep_id or tune_config.sweep_id,
        mf_model_src=mf_model_src,
        src_model_weights_path=src_model_weights_path,
        sim_matrix=dataset.sim_matrix,
        n_sh_users=dataset.n_sh_users,
        save_dir_path=save_dir_path,
        src_dataset_name=src_dataset_name,
        tgt_dataset_name=tgt_dataset_name,
        tgt_sparsity=tgt_sparsity,
        sweep_name=sweep_name,
    )


def test_ltn_reg(
    dataset: Dataset,
    config: ModelConfig,
    src_dataset_name: Domains_Type,
    tgt_dataset_name: Domains_Type,
    src_sparsity: float,
    user_level_src: bool,
    tgt_sparsity: float,
    save_dir_path: Path,
):
    train_config = config.ltn_reg_train_config

    tr, _, _ = _create_trainer(
        dataset=dataset,
        config=config,
        src_dataset_name=src_dataset_name,
        tgt_dataset_name=tgt_dataset_name,
        src_sparsity=src_sparsity,
        tgt_sparsity=tgt_sparsity,
        save_dir_path=save_dir_path,
        user_level_src=user_level_src,
    )

    weights_path = train_config.final_model_save_path
    tr.model.load_model(weights_path)

    te_loader = ValDataLoader(
        data=dataset.tgt_te, ui_matrix=dataset.tgt_ui_matrix, batch_size=train_config.hyper_params.batch_size
    )
    te_metric_results, _ = tr.validate(te_loader, val_metric=config.val_metric)
    logger.info(f"Test {config.val_metric.name}: {te_metric_results:.4f}")
