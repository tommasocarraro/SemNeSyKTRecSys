from typing import Any, Literal, Optional

import wandb
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch.optim import AdamW

from src.utils import set_seed
from .data_loader import DataLoader
from .loss import BPRLoss
from .metrics import Valid_Metrics_Type
from .model import MatrixFactorization
from .trainer import MfTrainer


def mf_tuning(
    seed: int,
    tune_config: dict[str, Any],
    train_set: NDArray,
    val_set: NDArray,
    val_batch_size: int,
    n_users: int,
    n_items: int,
    ui_matrix: csr_matrix,
    metric: Valid_Metrics_Type,
    n_epochs: Optional[int] = 1000,
    early: Optional[int] = 5,
    early_stopping_criterion: Literal["val_loss", "val_metric"] = "val_loss",
    entity_name: Optional[str] = None,
    exp_name: Optional[str] = None,
    bayesian_run_count: Optional[int] = 10,
    sweep_id: Optional[str] = None,
):
    """
    It performs the hyperparameter tuning of the MF model using the given hyperparameter search configuration,
    training and validation set. It can be used for both the MF model trained on the source domain and the baseline MF.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the tuning of hyperparameters
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param val_batch_size: batch size for validation
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param ui_matrix: sparse matrix of user interactions
    :param metric: validation metric that has to be used
    :param n_epochs: number of epochs for hyperparameter tuning
    :param early: number of epochs for early stopping
    :param early_stopping_criterion: whether to use the loss function or the validation metric as early stopping criterion
    :param entity_name: name of entity which owns the wandb project
    :param exp_name: name of experiment. It is used to log data to the corresponding WandB project
    :param bayesian_run_count: number of runs of Bayesian optimization
    :param sweep_id: sweep id if ones wants to continue a WandB that got blocked
    """
    set_seed(seed)
    # create loader for validation
    val_loader = DataLoader(val_set, ui_matrix, val_batch_size)

    # define function to call for performing one run of the hyperparameter search

    def tune():
        with wandb.init(project=exp_name, entity=entity_name) as run:
            # get one random configuration
            k = wandb.config.n_factors
            lr = wandb.config.learning_rate
            wd = wandb.config.weight_decay
            tr_batch_size = wandb.config.batch_size
            # set run name
            run_name = f"k={k}_lr={lr}_wd={wd}_bs={tr_batch_size}"
            run.name = run_name
            # define loader, model, optimizer and trainer
            train_loader = DataLoader(train_set, ui_matrix, tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k)
            optimizer = AdamW(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MfTrainer(mf, optimizer, loss=BPRLoss(), wandb_train=True)
            # perform training
            trainer.train(
                train_loader,
                val_loader,
                metric,
                n_epochs=n_epochs,
                early=early,
                verbose=1,
                early_stopping_criterion=early_stopping_criterion,
            )

    # launch the WandB sweep for 150 runs
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=tune_config, entity=entity_name, project=exp_name)
    wandb.agent(sweep_id, entity=entity_name, function=tune, count=bayesian_run_count, project=exp_name)
