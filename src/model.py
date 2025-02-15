import os
from pathlib import Path

import torch

from src.device import device
from loguru import logger


class MatrixFactorization(torch.nn.Module):
    """
    Vanilla Matrix factorization model.

    The model uses the inner product between a user and an item latent factor to get the prediction.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    The model has inside two vectors: one containing the biases of the users of the system, one containing the biases
    of the items of the system.
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int, normalize: bool = False):
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
        self.global_bias = torch.nn.Parameter(torch.rand(1))
        self.normalize = normalize
        # initialization with Glorot
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        torch.nn.init.xavier_normal_(self.i_emb.weight)
        torch.nn.init.xavier_normal_(self.u_bias.weight)
        torch.nn.init.xavier_normal_(self.i_bias.weight)

    def re_init_weights(self):
        """
        Reinitialize the weights of the model
        """
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        torch.nn.init.xavier_normal_(self.i_emb.weight)
        torch.nn.init.xavier_normal_(self.u_bias.weight)
        torch.nn.init.xavier_normal_(self.i_bias.weight)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor, dim: int = 1):
        """
        It computes the scores for the given user-item pairs using inner product (dot product).

        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        # add user and item biases to the prediction
        pred = pred + self.u_bias(u_idx) + self.i_bias(i_idx) + self.global_bias
        pred = pred.squeeze()
        if self.normalize:
            pred = torch.sigmoid(pred)
        return pred

    def save_model(self, path: Path):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        os.makedirs(path.parent, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path: Path):
        """
        Method for loading the model.

        :param path: path from which the final model has to be loaded.
        """
        if not path.is_file():
            logger.error(f"Model file '{path}' does not exist.")
            exit(1)
        final_model = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(final_model)
