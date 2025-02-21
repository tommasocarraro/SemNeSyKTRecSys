import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
import wandb
from loguru import logger
from ltn import LTNObject
from torch import Tensor
from torch.optim import Optimizer

from src.data_loader import DataLoader, ValDataLoader
from src.device import device
from src.metrics import PredictionMetricsType, RankingMetricsType, Valid_Metrics_Type, compute_metric
from src.model import MatrixFactorization
from src.utils import set_seed


class Trainer(ABC):
    model: MatrixFactorization
    optimizer: Optimizer
    loss: Union[
        Callable[[Tensor, Tensor], Tensor],
        Callable[[Tensor], Tensor],
        Callable[[LTNObject, LTNObject], LTNObject],
        Callable[[LTNObject], LTNObject],
    ]
    wandb_train: bool

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Union[DataLoader, ValDataLoader],
        val_metric: Valid_Metrics_Type,
        checkpoint_save_path: Optional[Path] = None,
        final_model_save_path: Optional[Path] = None,
        n_epochs: int = 500,
        early: Optional[int] = None,
        early_stopping_criterion: Literal["val_loss", "val_metric"] = "val_loss",
        verbose: int = 10,
    ):
        """
        Method for the train of the model.

        :param train_loader: data loader for training dataset
        :param val_loader: data loader for validation dataset
        :param val_metric: validation metric name
        :param checkpoint_save_path: Path where to save the training checkpoints
        :param final_model_save_path: Path where to save the final model
        :param n_epochs: number of epochs of training, default to 500
        :param early: patience for early stopping, default to None
        :param early_stopping_criterion: whether to use the loss function or the validation metric as early stopping criterion
        :param verbose: number of epochs to wait for printing training details
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
            try:
                # training step
                train_loss, log_dict = self.train_epoch(train_loader, epoch)
            except RuntimeError as e:
                logger.error(str(e))
                logger.error("Stopping training due to an error")
                break
            # validation step
            val_score, val_loss_dict = self.validate(val_loader, val_metric, use_val_loss=True)
            # merge log dictionaries
            log_dict.update(val_loss_dict)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = f"Epoch {epoch + 1} - Train loss {train_loss:.3f} - Val {val_metric.value} {val_score:.3f}"
                # add log_dict to log_record
                log_record += " - " + " - ".join([f"{k} {v:.3f}" for k, v in log_dict.items() if k != "train_loss"])
                # print epoch report
                logger.info(log_record)
                if self.wandb_train:
                    # log validation metric value
                    wandb.log({"Val metric": val_score})
                    # log training information
                    wandb.log(log_dict)
            # stop the training if vanishing or exploding gradients are detected
            if np.isnan(train_loss):
                logger.info("Training interrupted due to exploding or vanishing gradients")
                break
            # save best model and update early stop counter, if necessary
            if (val_score > best_val_score and not early_loss_based) or (
                early_loss_based and val_loss_dict["Val loss"] < best_val_score
            ):
                best_val_score = val_score if not early_loss_based else val_loss_dict["Val loss"]
                if self.wandb_train:
                    # the metric is logged only when a new best value is achieved for it
                    wandb.log(
                        {"Best val metric": val_score}
                        if not early_loss_based
                        else {"Best val loss": val_loss_dict["Val loss"]}
                    )
                early_counter = 0
                if checkpoint_save_path is not None:
                    logger.info(f"Saving checkpoint")
                    self.model.save_model(checkpoint_save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    logger.info("Training interrupted due to early stopping")
                    if final_model_save_path is not None and checkpoint_save_path is not None:
                        self.model.load_model(checkpoint_save_path)
                        self.model.save_model(final_model_save_path)
                        os.remove(checkpoint_save_path)
                    break

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
        return np.concatenate(users_), np.concatenate(pos_preds), np.concatenate(neg_preds)

    def prepare_for_evaluation_ranking(self, loader: ValDataLoader):
        """
        It prepares an array of predictions and targets for computing classification metrics.
        This is the function for the preparation for computation of ranking metrics.

        :param loader: loader containing evaluation data
        :return: predictions and targets
        """
        preds = []
        for batch_idx, (users, pos_items, neg_items) in enumerate(loader):
            pos_preds = self.predict(users, pos_items).cpu().numpy()
            neg_preds = (
                self.predict(users.repeat_interleave(neg_items.shape[1]), neg_items.flatten())
                .reshape(users.shape[0], -1)
                .cpu()
                .numpy()
            )
            preds.append(np.hstack((pos_preds.reshape(-1, 1), neg_preds)))

        return np.concatenate(preds)

    def compute_validation_loss(self, pos_preds: Tensor, neg_preds: Tensor):
        """
        Method for computing the validation loss for the model.

        :param pos_preds: predictions for positive interactions in the validation set
        :param neg_preds: predictions for negative interactions in the validation set
        :return: the validation loss for the model
        """
        return self.loss(pos_preds, neg_preds)

    def validate(
        self, val_loader: Union[DataLoader, ValDataLoader], val_metric: Valid_Metrics_Type, use_val_loss: bool = False
    ):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name (it is F0.5-score for source domain and F1-score for target domain)
        :param use_val_loss: whether to compute the validation loss or not
        :return: validation score based on the given metric averaged across all validation examples
        """
        if isinstance(val_metric, RankingMetricsType):
            if isinstance(val_loader, ValDataLoader):
                return self._validate_ranking(val_loader, val_metric)
            else:
                raise RuntimeError("A ValDataLoader was expected but got DataLoader instead")
        elif isinstance(val_metric, PredictionMetricsType):
            if isinstance(val_loader, DataLoader):
                return self._validate_preds(val_loader, val_metric, use_val_loss)
            else:
                raise RuntimeError("A DataLoader was expected but got ValDataLoader instead")
        raise ValueError("Unknown validation metric")

    def _validate_preds(self, val_loader: DataLoader, val_metric: PredictionMetricsType, use_val_loss: bool = False):
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

    def _validate_ranking(self, val_loader: ValDataLoader, val_metric: RankingMetricsType):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name (it is F0.5-score for source domain and F1-score for target domain)
        :return: validation score based on the given metric averaged across all validation examples
        """
        # prepare predictions and targets for evaluation
        preds = self.prepare_for_evaluation_ranking(val_loader)
        # compute validation metric
        val_score = compute_metric(metric=val_metric, preds=preds)

        return np.mean(val_score), {}

    def validate_k_times(
        self,
        k: int,
        initial_seed: int,
        train_loader: DataLoader,
        val_loader: Union[DataLoader, ValDataLoader],
        val_metric: Valid_Metrics_Type,
        checkpoint_save_path: Optional[Path] = None,
        final_model_save_path: Optional[Path] = None,
        n_epochs: int = 500,
        early: Optional[int] = None,
        early_stopping_criterion: Literal["val_loss", "val_metric"] = "val_loss",
        verbose: int = 10,
    ):
        """
        Method for the validating the model using k different seeds.

        :param k: the number of seeds to use
        :param initial_seed: the seed to use to randomly generate the k seeds
        :param train_loader: data loader for training dataset
        :param val_loader: data loader for validation dataset
        :param val_metric: validation metric name
        :param checkpoint_save_path: Path where to save the training checkpoints
        :param final_model_save_path: Path where to save the final model
        :param n_epochs: number of epochs of training, default to 500
        :param early: patience for early stopping, default to None
        :param early_stopping_criterion: whether to use the loss function or the validation metric as early stopping criterion
        :param verbose: number of epochs to wait for printing training details
        """
        # setting the initial seed to reproducibly create the list of seeds
        set_seed(initial_seed)
        # drawing k random integers between 0 and the maximum integer which numpy can store
        seeds = np.random.random_integers(low=0, high=np.iinfo(np.int32).max, size=k)
        val_results = []
        for seed in seeds:
            # resetting the run by using the new seed and initializing the model's weights
            set_seed(seed)
            self.model.re_init_weights()
            # training with the new initial weights
            self.train(
                train_loader=train_loader,
                val_loader=val_loader,
                val_metric=val_metric,
                checkpoint_save_path=checkpoint_save_path,
                final_model_save_path=final_model_save_path,
                n_epochs=n_epochs,
                early=early,
                early_stopping_criterion=early_stopping_criterion,
                verbose=verbose,
            )
            val_score, _ = self.validate(val_loader, val_metric, use_val_loss=True)
            val_results.append(val_score)
        return np.mean(val_results)

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        pass
