from typing import Any, Optional

import wandb
from numpy.typing import NDArray
from torch.optim import AdamW

from src.bpr_loss import BPRLoss
from src.loader import DataLoader
from src.metrics import Valid_Metrics_Type
from src.models.mf import MFTrainer, MatrixFactorization
from src.utils import set_seed
from scipy.sparse import csr_matrix


# TODO run should be named with hyperparameter values like in previous repository
def mf_tuning(
    seed: int,
    tune_config: dict[str, Any],
    train_set: NDArray,
    val_set: NDArray,
    n_users: int,
    n_items: int,
    ui_matrix: csr_matrix,
    metric: Valid_Metrics_Type,
    entity_name: Optional[str] = None,
    exp_name: Optional[str] = None,
    sweep_id: Optional[str] = None,
):
    """
    It performs the hyperparameter tuning of the MF model using the given hyperparameter search configuration,
    training and validation set. It can be used for both the MF model trained on the source domain and the baseline MF.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the tuning of hyperparameters
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param ui_matrix: sparse matrix of user interactions
    :param metric: validation metric that has to be used
    :param entity_name: name of entity which owns the wandb project
    :param exp_name: name of experiment. It is used to log data to the corresponding WandB project
    :param sweep_id: sweep id if ones wants to continue a WandB that got blocked
    """
    # create loader for validation
    val_loader = DataLoader(val_set, ui_matrix, 256)

    # define function to call for performing one run of the hyperparameter search

    def tune():
        with wandb.init(project=exp_name, entity=entity_name) as run:
            # get one random configuration
            k = wandb.config.k
            lr = wandb.config.lr
            wd = wandb.config.wd
            tr_batch_size = wandb.config.tr_batch_size
            # set run name
            run_name = f"k={k}_lr={lr}_wd={wd}_bs={tr_batch_size}"
            run.name = run_name
            # define loader, model, optimizer and trainer
            train_loader = DataLoader(train_set, ui_matrix, tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k)
            optimizer = AdamW(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MFTrainer(mf, optimizer, loss=BPRLoss(), wandb_train=True)
            # perform training
            trainer.train(
                train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1
            )

    # launch the WandB sweep for 150 runs
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=tune_config, entity=entity_name, project=exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, entity=entity_name, function=tune, count=20, project=exp_name)
