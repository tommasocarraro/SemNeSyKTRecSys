import ltn
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch.optim import Optimizer
from tqdm import tqdm

from src.cross_domain.loss import LTNLoss
from src.cross_domain.utils import sample_neg_items
from src.data_loader import DataLoader
from src.device import device
from src.model import MatrixFactorization
from src.trainer import Trainer


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
        processed_interactions: dict[int, NDArray],
        tgt_ui_matrix: csr_matrix,
        p_sat_agg: int,
        p_forall_ax1: int,
        p_forall_ax2: int,
        wandb_train=False,
    ):
        """
        Constructor of the trainer for the basic Logic Tensor Network model.

        :param mf_model: Matrix Factorization model to implement the Score function.
        :param optimizer: optimizer used for the training of the model
        :param processed_interactions: user-item interactions for which the sampling for the regularization axiom has
        to be performed
        :param tgt_ui_matrix: target domain user-item interactions
        :param p_sat_agg: hyperparameter p for sat aggregator
        :param p_forall_ax1: hyperparameter p for universal quantifier aggregator of axiom 1
        :param p_forall_ax2: hyperparameter p for universal quantifier aggregator of axiom 2
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
        self.p_forall_ax1 = p_forall_ax1
        self.p_forall_ax2 = p_forall_ax2
        self.tgt_ui_matrix = tgt_ui_matrix
        # array containing all users shared with at least one path found between the two domains
        self.sh_users = np.array(list(self.processed_interactions))

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
                p=self.p_forall_ax1,
            )
            ax1_sat.append(axiom1.value.item())

            # axiom 2
            # sample a number of shared users equal to the batch size
            sampled_sh_users = np.random.choice(self.sh_users, size=u.shape[0], replace=False)
            # sample positive interactions for all users among the connected items
            sampled_pos_items = [np.random.choice(self.processed_interactions[user]) for user in sampled_sh_users]
            # sample negative interactions for all users
            sampled_neg_items = sample_neg_items(
                sampled_sh_users=sampled_sh_users,
                tgt_ui_matrix=self.tgt_ui_matrix,
                processed_interactions=self.processed_interactions,
            )
            reg_user = ltn.Variable("reg_user", torch.tensor(sampled_sh_users), add_batch_dim=False)
            reg_pos_item = ltn.Variable("reg_pos_item", torch.tensor(sampled_pos_items), add_batch_dim=False)
            reg_neg_item = ltn.Variable("reg_neg_item", torch.tensor(sampled_neg_items), add_batch_dim=False)
            axiom2 = self.Forall(
                ltn.diag(reg_user, reg_pos_item, reg_neg_item),
                self.loss(self.Score(reg_user, reg_pos_item), self.Score(reg_user, reg_neg_item)),
                p=self.p_forall_ax2,
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
