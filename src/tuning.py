from src.loader import DataLoader
from src.models.mf import MatrixFactorization, MFTrainer
from torch.optim import AdamW
import wandb
from src.utils import set_seed
from src.bpr_loss import BPRLoss


# TODO run should be named with hyper-parameter values like in previous repository
def mf_tuning(seed, tune_config, train_set, val_set, n_users, n_items, metric, exp_name=None, sweep_id=None):
    """
    It performs the hyper-parameter tuning of the MF model using the given hyper-parameter search configuration,
    training and validation set. It can be used for both the MF model trained on the source domain and the baseline MF.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the tuning of hyper-parameters
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param metric: validation metric that has to be used
    :param exp_name: name of experiment. It is used to log data to the corresponding WandB project
    :param sweep_id: sweep id if ones wants to continue a WandB that got blocked
    """
    # create loader for validation
    val_loader = DataLoader(val_set, n_items, 256)

    # define function to call for performing one run of the hyper-parameter search

    def tune():
        with wandb.init(project=exp_name) as run:
            # get one random configuration
            k = wandb.config.k
            lr = wandb.config.lr
            wd = wandb.config.wd
            tr_batch_size = wandb.config.tr_batch_size
            # set run name
            run_name = f"k={k}_lr={lr}_wd={wd}_bs={tr_batch_size}"
            run.name = run_name
            # define loader, model, optimizer and trainer
            train_loader = DataLoader(train_set, n_items, tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k)
            optimizer = AdamW(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MFTrainer(mf, optimizer, loss=BPRLoss(), wandb_train=True)
            # perform training
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for 150 runs
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=tune_config, project=exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, function=tune, count=20, project=exp_name)
