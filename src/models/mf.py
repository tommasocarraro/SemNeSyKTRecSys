import torch
from src.trainer import Trainer
from tqdm import tqdm


class MatrixFactorization(torch.nn.Module):
    """
    Vanilla Matrix factorization model.

    The model uses the inner product between a user and an item latent factor to get the prediction.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    The model has inside two vectors: one containing the biases of the users of the system, one containing the biases
    of the items of the system.
    """
    def __init__(self, n_users, n_items, n_factors, normalize=False):
        """
        Constructor of the matrix factorization model.

        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings for users and items
        :param normalize: whether the output has to be normalized in [0.,1.] using sigmoid. This is used for the LTN
        model.
        """
        super(MatrixFactorization, self).__init__()
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

    def forward(self, u_idx, i_idx, dim=1):
        """
        It computes the scores for the given user-item pairs using inner product (dot product).

        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        # add user and item biases to the prediction
        pred += self.u_bias(u_idx) + self.i_bias(i_idx)
        pred = pred.squeeze()
        if self.normalize:
            pred = torch.sigmoid(pred)
        return pred


class MFTrainer(Trainer):
    """
    Basic trainer for training a Matrix Factorization model using gradient descent.
    """
    def __init__(self, mf_model, optimizer, loss, wandb_train=False):
        """
        Constructor of the trainer for the MF model.

        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for training the Matrix Factorization model. It could be MSE or Focal Loss
        """
        super(MFTrainer, self).__init__(mf_model, optimizer, wandb_train)
        self.loss = loss

    def train_epoch(self, train_loader, epoch=None):
        train_loss = 0.0
        for batch_idx, (u_i_pairs, ratings) in tqdm(enumerate(train_loader)):
            self.optimizer.zero_grad()
            loss = self.loss(self.model(u_i_pairs[:, 0], u_i_pairs[:, 1]), ratings)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {"train_loss": train_loss / len(train_loader)}


class MFTrainerClassifier(MFTrainer):
    """
    Trainer for training a Matrix Factorization model for the binary classification task.

    This version of the trainer uses the focal loss as loss function. It implements a classification task, where the
    objective is to minimize the focal loss. The ratings are binary, hence they can be interpreted as binary classes.
    The objective is to discriminate between class 1 ("likes") and class 0 ("dislikes"). Note the focal loss is a
    generalization of the binary cross-entropy to deal with imbalance data.
    """
    def __init__(self, mf_model, optimizer, loss, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the MF model for binary classification.

        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param loss: loss function used to train the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param threshold: threshold used to determine whether an example is negative or positive (decision boundary)
        """
        super(MFTrainerClassifier, self).__init__(mf_model, optimizer, loss, wandb_train)
        self.threshold = threshold

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        # apply sigmoid because during training the losses with logits are used for numerical stability
        preds = super().predict(x, dim)
        preds = torch.sigmoid(preds)
        preds = preds >= self.threshold
        return preds
