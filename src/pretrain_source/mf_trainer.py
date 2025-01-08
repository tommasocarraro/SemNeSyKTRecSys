from typing import Callable, Union

from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from src.data_loader import DataLoader
from src.device import device
from src.model import MatrixFactorization
from src.trainer import Trainer


class MfTrainer(Trainer):
    """
    Basic trainer for training a Matrix Factorization model using gradient descent.
    """

    def __init__(
        self,
        model: MatrixFactorization,
        optimizer: Optimizer,
        loss: Union[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]],
        wandb_train: bool = False,
    ):
        """
        Constructor of the trainer for the MF model.

        :param model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for training the Matrix Factorization model. It could be MSE or Focal Loss
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.wandb_train = wandb_train
        self.loss = loss

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :param epoch: current epoch
        :return: training loss value averaged across training batches and a dictionary containing useful information
        to log, such as other metrics computed by this model
        """
        train_loss = 0.0
        for batch_idx, (user, pos_items, neg_items) in enumerate(
            tqdm(train_loader, desc=f"Training epoch {epoch}", dynamic_ncols=True)
        ):
            self.optimizer.zero_grad()
            pos_preds = self.model(user, pos_items)
            neg_preds = self.model(user, neg_items)
            loss = self.loss(pos_preds, neg_preds)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {"train_loss": train_loss / len(train_loader)}
