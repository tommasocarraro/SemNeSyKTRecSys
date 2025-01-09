import ltn
import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim import Optimizer
from tqdm import tqdm

from src.data_loader import DataLoader
from src.model import MatrixFactorization
from src.cross_domain.loss import LTNLoss
from src.trainer import Trainer
from src.device import device


class LTNTrainer(Trainer):
    """
    Trainer for the training of a Logic Tensor Network model with Matrix Factorization as the predictor. This model
    simply implements the Matrix Factorization using a Logic Tensor Network. The loss function is similar to the BPR
    loss but constrained to be in the range [0., 1.] to work with the LTN.
    """

    def __init__(self, mf_model: MatrixFactorization, optimizer: Optimizer, p_forall=2, wandb_train=False):
        """
        Constructor of the trainer for the basic Logic Tensor Network model.

        :param mf_model: Matrix Factorization model to implement the Score function.
        :param optimizer: optimizer used for the training of the model
        :param p_forall: hyperparameter p for universal quantifier aggregator
        :param wandb_train: whether the training information has to be logged on WandB or not
        """
        self.model = mf_model.to(device)
        self.optimizer = optimizer
        self.wandb_train = wandb_train
        self.loss = LTNLoss()
        self.Score = ltn.Function(mf_model)
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier="f")
        self.p_forall = p_forall

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        train_loss, train_sat_agg = 0.0, 0.0
        for batch_idx, (u, i_pos, i_neg) in enumerate(
            tqdm(train_loader, desc=f"Training epoch {epoch}", dynamic_ncols=True)
        ):
            self.optimizer.zero_grad()
            user = ltn.Variable("user", u, add_batch_dim=False)
            item_pos = ltn.Variable("item_pos", i_pos, add_batch_dim=False)
            item_neg = ltn.Variable("item_neg", i_neg, add_batch_dim=False)
            train_sat = self.Forall(
                ltn.diag(user, item_pos, item_neg),
                self.loss(self.Score(user, item_pos), self.Score(user, item_neg)),
                p=self.p_forall,
            ).value
            train_sat_agg += train_sat.item()
            loss = 1.0 - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader)}


class LTNRegTrainer(Trainer):
    """
    Trainer for the training of a Logic Tensor Network model with Matrix Factorization as the predictor. This model adds
    a regularization axiom to the previous one, which allows the transfer of knowledge from source to target domain
    """

    def __init__(
        self,
        mf_model: MatrixFactorization,
        optimizer: Optimizer,
        processed_interactions: NDArray,
        p_sat_agg: int,
        p_forall: int,
        neg_score_value: float,
        wandb_train=False,
    ):
        """
        Constructor of the trainer for the basic Logic Tensor Network model.

        :param mf_model: Matrix Factorization model to implement the Score function.
        :param optimizer: optimizer used for the training of the model
        :param processed_interactions: user-item interactions for which the sampling for the regularization axiom has
        to be performed
        :param p_sat_agg: hyperparameter p for sat aggregator
        :param p_forall: hyperparameter p for universal quantifier aggregator
        :param wandb_train: whether the training information has to be logged on WandB or not
        """
        self.model = mf_model.to(device)
        self.optimizer = optimizer
        self.loss = LTNLoss()
        self.wandb_train = wandb_train
        self.processed_interactions = processed_interactions
        self.Score = ltn.Function(mf_model)
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier="f")
        self.SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=p_sat_agg))
        self.p_forall = p_forall
        self.neg_score_value = neg_score_value

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        train_loss, train_sat_agg, ax1_sat, ax2_sat = 0.0, 0.0, [], []
        for batch_idx, (u, i_pos, i_neg) in enumerate(
            tqdm(train_loader, desc=f"Training epoch {epoch}", dynamic_ncols=True)
        ):
            self.optimizer.zero_grad()
            # axiom 1
            user = ltn.Variable("user", u, add_batch_dim=False)
            item_pos = ltn.Variable("item_pos", i_pos, add_batch_dim=False)
            item_neg = ltn.Variable("item_neg", i_neg, add_batch_dim=False)
            axiom1 = self.Forall(
                ltn.diag(user, item_pos, item_neg),
                self.loss(self.Score(user, item_pos), self.Score(user, item_neg)),
                p=self.p_forall,
            )
            ax1_sat.append(axiom1.value.item())
            # axiom 2
            # sample interactions from the list of processed interactions
            proc_int_indices = np.random.randint(0, len(self.processed_interactions), u.shape[0])
            batch_proc_int = self.processed_interactions[proc_int_indices]
            # this is a negative score that is fixed during training. Every time the model transfer knowledge, it has to
            # maximize the distance of the positive item score from this score. This might be a hyper-parameter
            neg_score = ltn.Variable(
                "neg_score", torch.tensor([self.neg_score_value] * u.shape[0]), add_batch_dim=False
            )
            # define variables
            reg_user = ltn.Variable("reg_user", torch.tensor(batch_proc_int[:, 0]), add_batch_dim=False)
            reg_item = ltn.Variable("reg_item", torch.tensor(batch_proc_int[:, 2]), add_batch_dim=False)
            # define axiom 2
            axiom2 = self.Forall(
                ltn.diag(reg_user, reg_item, neg_score),
                self.loss(self.Score(reg_user, reg_item), neg_score),
                p=self.p_forall,
            )
            ax2_sat.append(axiom2.value.item())
            train_sat = self.SatAgg(axiom1, axiom2)
            train_sat_agg += train_sat.item()
            loss = 1.0 - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {
            "training_overall_sat": train_sat_agg / len(train_loader),
            "axiom1_sat": np.mean(ax1_sat),
            "axiom2_sat": np.mean(ax2_sat),
        }
