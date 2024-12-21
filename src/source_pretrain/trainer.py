import os
import sys
from typing import Callable, Optional, Literal

import numpy as np
import torch
import wandb
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from src.device import device
from .data_loader import DataLoader, ValDataLoader
from .metrics import Valid_Metrics_Type, compute_metric
from .model import MatrixFactorization
from pathlib import Path


class MfTrainer:
    """
    Basic trainer for training a Matrix Factorization model using gradient descent.
    """

    def __init__(
            self,
            model: MatrixFactorization,
            optimizer: torch.optim.Optimizer,
            loss: Callable[[Tensor, Tensor], Tensor],
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

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            val_metric: Valid_Metrics_Type,
            n_epochs: int = 500,
            early: Optional[int] = None,
            early_stopping_criterion: Literal["val_loss", "val_metric"] = "val_loss",
            verbose: int = 10,
            save_paths: Optional[tuple[Path, Path]] = None,
    ):
        """
        Method for the train of the model.

        :param train_loader: data loader for training dataset
        :param val_loader: data loader for validation dataset
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 500
        :param early: patience for early stopping, default to None
        :param early_stopping_criterion: whether to use the loss function or the validation metric as early stopping criterion
        :param verbose: number of epochs to wait for printing training details
        :param save_paths: tuple of paths where to save the checkpoint (first path) and the best model (second path)
        """
        logger.debug(f"Starting training on {device}")
        early_loss_based = True if early_stopping_criterion == "val_loss" else False
        if early_loss_based:
            best_val_score = sys.maxsize
        else:
            best_val_score = 0.0
        early_counter = 0
        if self.wandb_train:
            # log gradients and parameters with Weights and Biases
            wandb.watch(self.model, log="all")

        for epoch in range(n_epochs):
            # training step
            train_loss, log_dict = self.train_epoch(train_loader)
            # validation step
            val_score, val_loss_dict = self.validate(
                val_loader, val_metric, use_val_loss=True
            )
            # merge log dictionaries
            log_dict.update(val_loss_dict)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = f"Epoch {epoch + 1} - Train loss {train_loss:.3f} - Val {val_metric} {val_score:.3f}"
                # add log_dict to log_record
                log_record += " - " + " - ".join(
                    [f"{k} {v:.3f}" for k, v in log_dict.items() if k != "train_loss"]
                )
                # print epoch report
                logger.info(log_record)
                if self.wandb_train:
                    # log validation metric value
                    wandb.log({"Val %s" % (val_metric,): val_score})
                    # log training information
                    wandb.log(log_dict)
            # stop the training if vanishing or exploding gradients are detected
            if np.isnan(train_loss):
                logger.info(
                    "Training interrupted due to exploding or vanishing gradients"
                )
                break
            # save best model and update early stop counter, if necessary
            if (val_score > best_val_score and not early_loss_based) or (
                    val_loss_dict["Val loss"] < best_val_score and early_loss_based
            ):
                best_val_score = (
                    val_score if not early_loss_based else val_loss_dict["Val loss"]
                )
                if self.wandb_train:
                    # the metric is logged only when a new best value is achieved for it
                    wandb.log(
                        {"Best val %s" % (val_metric,): val_score}
                        if not early_loss_based
                        else {"Best val loss": val_loss_dict["Val loss"]}
                    )
                early_counter = 0
                if save_paths:
                    logger.info(f"Saving checkpoint")
                    self.save_model(save_paths[0], is_checkpoint=True)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    logger.info("Training interrupted due to early stopping")
                    if save_paths:
                        self.load_model(save_paths[0])
                        self.save_model(save_paths[1], is_checkpoint=False)
                        os.remove(save_paths[0])
                    break

    def train_epoch(self, train_loader: DataLoader):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
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

    def predict(self, users: Tensor, items: Tensor, dim: int = 1):
        """
        Method for performing a prediction of the model for given user-item pairs.

        :param users: tensor containing user indices for which predictions need to be made
        :param items: tensor containing item indices corresponding to the users
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: tensor with the prediction scores for the user-item pairs
        """
        with torch.no_grad():
            # Call the model with users and items to get predicted scores
            return self.model(users, items, dim)

    def prepare_for_evaluation(self, loader: DataLoader):
        """
        It prepares an array of predictions and targets for computing classification metrics.

        :param loader: loader containing evaluation data
        :return: predictions and targets
        """
        users_, pos_preds, neg_preds = [], [], []
        for batch_idx, (users, pos_items, neg_items) in enumerate(loader):
            users_.append(users.cpu().numpy())
            pos_preds.append(self.predict(users, pos_items).cpu().numpy())
            neg_preds.append(self.predict(users, neg_items).cpu().numpy())
        return (
            np.concatenate(users_),
            np.concatenate(pos_preds),
            np.concatenate(neg_preds),
        )

    def prepare_for_evaluation_ranking(self, loader: ValDataLoader):
        """
        It prepares an array of predictions and targets for computing classification metrics.
        This is the function for the preparation for computation of ranking metrics.

        :param loader: loader containing evaluation data
        :return: predictions and targets
        """
        preds, ground_truth = [], []
        gen = enumerate(loader)
        for batch_idx, (users, pos_items, neg_items, gt) in tqdm(gen, total=len(loader)):
            pos_preds = self.predict(users, pos_items).cpu().numpy()
            # neg_preds_ = []
            # for user, neg_items_ in tqdm(zip(users, neg_items)):
            #     neg_preds_.append(self.predict(user.repeat_interleave(neg_items.shape[1]),
            #                                    neg_items_).cpu().numpy())
            # neg_preds = np.stack(neg_preds_)
            # preds.append(np.hstack((pos_preds.reshape(-1, 1), neg_preds)))
            # ground_truth.append(gt)

            neg_preds = self.predict(users.repeat_interleave(neg_items.shape[1]), neg_items.flatten()).reshape(
                users.shape[0], -1).numpy()
            preds.append(np.hstack((pos_preds.reshape(-1, 1), neg_preds)))
            ground_truth.append(gt)

        return np.concatenate(preds), np.concatenate(ground_truth)

    def compute_validation_loss(self, pos_preds: Tensor, neg_preds: Tensor):
        """
        Method for computing the validation loss for the model.

        :param pos_preds: predictions for positive interactions in the validation set
        :param neg_preds: predictions for negative interactions in the validation set
        :return: the validation loss for the model
        """
        return self.loss(pos_preds, neg_preds)

    def validate(
            self,
            val_loader: DataLoader,
            val_metric: Valid_Metrics_Type,
            use_val_loss: bool = False,
    ):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name (it is F0.5-score for source domain and F1-score for target domain)
        :param use_val_loss: whether to compute the validation loss or not
        :return: validation score based on the given metric averaged across all validation examples
        """
        # prepare predictions and targets for evaluation
        users, pos_preds, neg_preds = self.prepare_for_evaluation(val_loader)
        # compute validation metric
        val_score = compute_metric(val_metric, pos_preds, neg_preds, users)
        # compute validation loss
        validation_loss = None
        if use_val_loss:
            validation_loss = self.compute_validation_loss(
                torch.tensor(pos_preds).to(device), torch.tensor(neg_preds).to(device)
            )

        return np.mean(val_score), {"Val loss": validation_loss}

    def validate_ranking(
            self,
            val_loader: ValDataLoader,
            val_metric: Valid_Metrics_Type,
    ):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name (it is F0.5-score for source domain and F1-score for target domain)
        :return: validation score based on the given metric averaged across all validation examples
        """
        # prepare predictions and targets for evaluation
        preds, ground_truth = self.prepare_for_evaluation_ranking(val_loader)
        # compute validation metric
        val_score = compute_metric(val_metric, preds, ground_truth)

        return np.mean(val_score)

    def save_model(self, path: Path, is_checkpoint=True):
        """
        Method for saving the model.

        :param path: path where to save the model
        :param is_checkpoint: whether the model to be saved is a checkpoint or the final one
        """
        os.makedirs(path.parent, exist_ok=True)
        if is_checkpoint:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path,
            )
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: Path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
