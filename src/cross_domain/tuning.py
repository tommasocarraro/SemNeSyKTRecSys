from pathlib import Path
from typing import Any, Literal, Optional

import wandb
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch.optim import AdamW

from src.cross_domain.ltn_trainer import LTNRegTrainer
from src.cross_domain.utils import get_reg_axiom_data
from src.data_loader import TrDataLoader, ValDataLoader
from src.metrics import PredictionMetricsType, RankingMetricsType, Valid_Metrics_Type
from src.model import MatrixFactorization
from src.pretrain_source.inference import generate_pre_trained_src_matrix


# def ltn_tuning(
#     tune_config: dict[str, Any],
#     train_set: NDArray,
#     val_set: NDArray,
#     val_batch_size: int,
#     n_users: int,
#     n_items: int,
#     tgt_ui_matrix: csr_matrix,
#     val_metric: Valid_Metrics_Type,
#     early_stopping_criterion: Literal["val_loss", "val_metric"],
#     n_epochs: Optional[int] = 1000,
#     early: Optional[int] = 5,
#     entity_name: Optional[str] = None,
#     exp_name: Optional[str] = None,
#     bayesian_run_count: Optional[int] = 10,
#     sweep_id: Optional[str] = None,
#     sweep_name: Optional[str] = None,
# ):
#     """
#     It performs the hyperparameter tuning of the MF model using the given hyperparameter search configuration,
#     training and validation set. It can be used for both the MF model trained on the source domain and the baseline MF.
#
#     :param tune_config: configuration for the tuning of hyperparameters
#     :param train_set: train set on which the tuning is performed
#     :param val_set: validation set on which the tuning is evaluated
#     :param val_batch_size: batch size for validation
#     :param n_users: number of users in the dataset
#     :param n_items: number of items in the dataset
#     :param tgt_ui_matrix: sparse matrix of user interactions
#     :param val_metric: validation metric that has to be used
#     :param n_epochs: number of epochs for hyperparameter tuning
#     :param early: number of epochs for early stopping
#     :param early_stopping_criterion: whether to use the loss function or the validation metric as early stopping criterion
#     :param entity_name: name of entity which owns the wandb project
#     :param exp_name: name of experiment. It is used to log data to the corresponding WandB project
#     :param bayesian_run_count: number of runs of Bayesian optimization
#     :param sweep_id: sweep id if ones wants to continue a WandB that got blocked
#     :param sweep_name: name to give to the sweep
#     """
#     # create loader for validation
#     if val_metric in RankingMetricsType:
#         val_loader = ValDataLoader(data=val_set, ui_matrix=tgt_ui_matrix, batch_size=val_batch_size)
#     elif val_metric in PredictionMetricsType:
#         val_loader = DataLoader(data=val_set, ui_matrix=tgt_ui_matrix, batch_size=val_batch_size)
#     else:
#         raise ValueError(f"{val_metric} is not a valid metric")
#
#     # define function to call for performing one run of the hyperparameter search
#
#     def tune():
#         with wandb.init() as run:
#             # get one random configuration
#             k = wandb.config.n_factors
#             lr = wandb.config.learning_rate
#             wd = wandb.config.weight_decay
#             tr_batch_size = wandb.config.batch_size
#             p_forall = wandb.config.p_forall
#             # set run name
#             run.name = f"k={k}_lr={lr}_wd={wd}_bs={tr_batch_size}_p_forall={p_forall}"
#             # define loader, model, optimizer and trainer
#             train_loader = DataLoader(train_set, tgt_ui_matrix, tr_batch_size)
#             mf = MatrixFactorization(n_users, n_items, k)
#             optimizer = AdamW(mf.parameters(), lr=lr, weight_decay=wd)
#             trainer = LTNTrainer(mf_model=mf, optimizer=optimizer, p_forall=p_forall, wandb_train=True)
#             # perform training
#             trainer.train(
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 val_metric=val_metric,
#                 n_epochs=n_epochs,
#                 early=early,
#                 verbose=1,
#                 early_stopping_criterion=early_stopping_criterion,
#             )
#
#     if sweep_name is not None:
#         tune_config["name"] = sweep_name
#
#     # launch the WandB sweep for 150 runs
#     if sweep_id is None:
#         sweep_id = wandb.sweep(sweep=tune_config, entity=entity_name, project=exp_name)
#     wandb.agent(sweep_id, function=tune, entity=entity_name, project=exp_name, count=bayesian_run_count)


def ltn_tuning_reg(
    tune_config: dict[str, Any],
    train_set: NDArray,
    val_set: NDArray,
    val_batch_size: int,
    n_users: int,
    n_sh_users: int,
    n_items: int,
    sim_matrix: csr_matrix,
    src_ui_matrix: csr_matrix,
    tgt_ui_matrix: csr_matrix,
    mf_model_src: MatrixFactorization,
    src_model_weights_path: Path,
    src_dataset_name: str,
    tgt_dataset_name: str,
    tgt_sparsity: float,
    save_dir_path: Path,
    val_metric: Valid_Metrics_Type,
    early_stopping_criterion: Literal["val_loss", "val_metric"],
    n_epochs: Optional[int] = 1000,
    early: Optional[int] = 5,
    entity_name: Optional[str] = None,
    exp_name: Optional[str] = None,
    bayesian_run_count: Optional[int] = 10,
    sweep_id: Optional[str] = None,
    sweep_name: Optional[str] = None,
):
    """
    It performs the hyperparameter tuning of the MF model using the given hyperparameter search configuration,
    training and validation set. It can be used for both the MF model trained on the source domain and the baseline MF.

    :param tune_config: configuration for the tuning of hyperparameters
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param val_batch_size: batch size for validation
    :param n_users: number of users in the dataset
    :param n_sh_users: number of users shared between the source and target domains
    :param n_items: number of items in the dataset
    :param sim_matrix: sparse matrix representing whether paths between source and target domain items exist
    :param src_ui_matrix: sparse matrix of user interactions from source domain
    :param tgt_ui_matrix: sparse matrix of user interactions from target domain
    :param src_dataset_name: label of the source domain, used for naming temporary files
    :param tgt_dataset_name: label of the target domain, used for naming temporary files
    :param tgt_sparsity: the sparsity factor of the target domain
    :param save_dir_path: directory where to store temporary files
    :param val_metric: validation metric that has to be used
    :param n_epochs: number of epochs for hyperparameter tuning
    :param early: number of epochs for early stopping
    :param early_stopping_criterion: whether to use the loss function or the validation metric as early stopping criterion
    :param entity_name: name of entity which owns the wandb project
    :param exp_name: name of experiment. It is used to log data to the corresponding WandB project
    :param bayesian_run_count: number of runs of Bayesian optimization
    :param sweep_id: sweep id if ones wants to continue a WandB that got blocked
    :param sweep_name: name to give to the sweep
    """
    # create loader for validation
    if val_metric in RankingMetricsType:
        val_loader = ValDataLoader(data=val_set, ui_matrix=tgt_ui_matrix, batch_size=val_batch_size)
    elif val_metric in PredictionMetricsType:
        val_loader = TrDataLoader(data=val_set, ui_matrix=tgt_ui_matrix, batch_size=val_batch_size)
    else:
        raise ValueError(f"{val_metric} is not a valid metric")

    # define function to call for performing one run of the hyperparameter search

    def tune():
        with wandb.init() as run:
            # get one random configuration
            k = wandb.config.n_factors
            lr = wandb.config.learning_rate
            wd = wandb.config.weight_decay
            tr_batch_size = wandb.config.batch_size
            p_forall_ax1 = wandb.config.p_forall_ax1
            p_forall_ax2 = wandb.config.p_forall_ax2
            p_sat_agg = wandb.config.p_sat_agg
            neg_score_value = wandb.config.neg_score_value
            top_k_src = wandb.config.top_k_src

            top_k_preds = generate_pre_trained_src_matrix(
                mf_model=mf_model_src,
                best_weights_path=src_model_weights_path,
                n_shared_users=n_sh_users,
                save_dir_path=save_dir_path,
                batch_size=2048,
                k=top_k_src,
            )
            processed_interactions = get_reg_axiom_data(
                src_ui_matrix=src_ui_matrix,
                tgt_ui_matrix=tgt_ui_matrix,
                n_sh_users=n_sh_users,
                sim_matrix=sim_matrix,
                top_k_items=top_k_preds,
                save_dir_path=save_dir_path,
                src_dataset_name=src_dataset_name,
                tgt_dataset_name=tgt_dataset_name,
                tgt_sparsity=tgt_sparsity,
            )
            # set run name
            run.name = f"k={k}_lr={lr}_wd={wd}_bs={tr_batch_size}_p_forall_ax1={p_forall_ax1}_p_forall_ax2={p_forall_ax2}_p_sat_agg={p_sat_agg}_neg_score_value={neg_score_value}"
            # define loader, model, optimizer and trainer
            train_loader = TrDataLoader(data=train_set, ui_matrix=tgt_ui_matrix, batch_size=tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k)
            optimizer = AdamW(mf.parameters(), lr=lr, weight_decay=wd)

            trainer = LTNRegTrainer(
                mf_model=mf,
                optimizer=optimizer,
                p_forall_ax1=p_forall_ax1,
                p_forall_ax2=p_forall_ax2,
                p_sat_agg=p_sat_agg,
                neg_score_value=neg_score_value,
                processed_interactions=processed_interactions,
                wandb_train=True,
            )
            # perform training
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                val_metric=val_metric,
                n_epochs=n_epochs,
                early=early,
                verbose=1,
                early_stopping_criterion=early_stopping_criterion,
            )

    if sweep_name is not None:
        tune_config["name"] = sweep_name

    # launch the WandB sweep for 50 runs
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=tune_config, entity=entity_name, project=exp_name)
    wandb.agent(sweep_id, function=tune, entity=entity_name, project=exp_name, count=bayesian_run_count)
