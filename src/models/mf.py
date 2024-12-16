from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from src.loader import DataLoader
from src.trainer import Trainer


class MatrixFactorization(torch.nn.Module):
    """
    Vanilla Matrix factorization model.

    The model uses the inner product between a user and an item latent factor to get the prediction.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    The model has inside two vectors: one containing the biases of the users of the system, one containing the biases
    of the items of the system.
    """

    def __init__(
        self, n_users: int, n_items: int, n_factors: int, normalize: bool = False
    ):
        """
        Constructor of the matrix factorization model.

        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings for users and items
        :param normalize: whether the output has to be normalized in [0.,1.] using sigmoid. This is used for the LTN
        model.
        """
        super(MatrixFactorization, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        self.u_bias = torch.nn.Embedding(n_users, 1)
        self.i_bias = torch.nn.Embedding(n_items, 1)
        self.normalize = normalize
        # initialization with Glorot
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        torch.nn.init.xavier_normal_(self.i_emb.weight)
        torch.nn.init.xavier_normal_(self.u_bias.weight)
        torch.nn.init.xavier_normal_(self.i_bias.weight)

    def forward(self, u_idx: Tensor, i_idx: Tensor, dim: int = 1):
        """
        It computes the scores for the given user-item pairs using inner product (dot product).

        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        # add user and item biases to the prediction
        pred = pred + self.u_bias(u_idx) + self.i_bias(i_idx)
        pred = pred.squeeze()
        if self.normalize:
            pred = torch.sigmoid(pred)
        return pred


class MFTrainer(Trainer):
    """
    Basic trainer for training a Matrix Factorization model using gradient descent.
    """

    def __init__(
        self,
        mf_model: MatrixFactorization,
        optimizer: Optimizer,
        loss: Callable[[Tensor, Tensor], Tensor],
        wandb_train: bool = False,
    ):
        """
        Constructor of the trainer for the MF model.

        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for training the Matrix Factorization model. It could be MSE or Focal Loss
        """
        super(MFTrainer, self).__init__(mf_model, optimizer, wandb_train)
        self.loss = loss

    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :param epoch: index of epoch
        :return: training loss value averaged across training batches and a dictionary containing useful information
        to log, such as other metrics computed by this model
        """
        train_loss = 0.0
        for batch_idx, (user, pos_items, neg_items) in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()
            pos_preds = self.model(user, pos_items)
            neg_preds = self.model(user, neg_items)
            loss = self.loss(pos_preds, neg_preds)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {
            "train_loss": train_loss / len(train_loader)
        }

    def compute_validation_loss(self, pos_preds: Tensor, neg_preds: Tensor):
        """
        Method for computing the validation loss for the model.

        :param pos_preds: predictions for positive interactions in the validation set
        :param neg_preds: predictions for negative interactions in the validation set
        :return: the validation loss for the model
        """
        return self.loss(pos_preds, neg_preds)
