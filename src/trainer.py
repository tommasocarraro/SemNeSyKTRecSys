from typing import Optional

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor

from src import device
from src.loader import DataLoader
from src.metrics import Valid_Metrics_Type, check_metrics, compute_metric


class Trainer:
    """
    Abstract base class that manages the training of the model.

    Each model implementation must inherit from this class and implement the train_epoch() method. It is also possible
    to use overloading to redefine the behavior of some methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        wandb_train: bool = False,
    ):
        """
        Constructor of the trainer.

        :param model: neural model for which the training has to be performed
        :param optimizer: optimizer that has to be used for the training of the model
        :param wandb_train: whether to log data on Weights and Biases servers. This is used when using hyperparameter
        optimization
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.wandb_train = wandb_train

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_metric: Valid_Metrics_Type,
        n_epochs: int = 500,
        early: Optional[int] = None,
        verbose: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Method for the train of the model.

        :param train_loader: data loader for training dataset
        :param val_loader: data loader for validation dataset
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 500
        :param early: patience for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details
        :param save_path: path where to save the best model, default to None
        """
        check_metrics(val_metric)
        best_val_score = 0.0
        early_counter = 0
        if self.wandb_train:
            # log gradients and parameters with Weights and Biases
            wandb.watch(self.model, log="all")

        for epoch in range(n_epochs):
            # training step
            train_loss, log_dict = self.train_epoch(train_loader, epoch + 1)
            # validation step
            val_score, val_loss_dict = self.validate(
                val_loader, val_metric, use_val_loss=True
            )
            # merge log dictionaries
            log_dict.update(val_loss_dict)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = "Epoch %d - Train loss %.3f - Val %s %.3f" % (
                    epoch + 1,
                    train_loss,
                    val_metric,
                    val_score,
                )
                # add log_dict to log_record
                log_record += " - " + " - ".join(
                    [
                        "%s %.3f" % (k, v)
                        for k, v in log_dict.items()
                        if k != "train_loss"
                    ]
                )
                # print epoch report
                print(log_record)
                if self.wandb_train:
                    # log validation metric value
                    wandb.log({"smooth_%s" % (val_metric,): val_score})
                    # log training information
                    wandb.log(log_dict)
            # stop the training if vanishing or exploding gradients are detected
            if np.isnan(train_loss):
                print("Training interrupted due to exploding or vanishing gradients")
                break
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                if self.wandb_train:
                    # the metric is logged only when a new best value is achieved for it
                    wandb.log({"%s" % (val_metric,): val_score})
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    if save_path:
                        self.load_model(save_path)
                    break

    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :param epoch: index of epoch
        :return: training loss value averaged across training batches and a dictionary containing useful information
        to log, such as other metrics computed by this model
        """
        raise NotImplementedError()

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

    def compute_validation_loss(self, pos_preds: Tensor, neg_preds: Tensor):
        """
        Method for computing the validation loss for the model.

        :param pos_preds: predictions for positive interactions in the validation set
        :param neg_preds: predictions for negative interactions in the validation set
        :return: the validation loss for the model
        """
        raise NotImplementedError()

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
        # # compute precision and recall
        # p, r, f, _ = precision_recall_fscore_support(
        #     targets,
        #     preds,
        #     beta=float(val_metric.split("-")[1]) if "fbeta" in val_metric else 1.0,
        #     average=None,
        # )
        # # compute other useful metrics used in classification tasks
        # tn, fp, fn, tp = tuple(confusion_matrix(targets, preds).ravel())
        # sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        # log metrics to WandB servers
        # if self.wandb_train:
        #     wandb.log(
        #         {
        #             "neg_prec": p[0],
        #             "pos_prec": p[1],
        #             "neg_rec": r[0],
        #             "pos_rec": r[1],
        #             "neg_f": f[0],
        #             "pos_f": f[1],
        #             "tn": tn,
        #             "fp": fp,
        #             "fn": fn,
        #             "tp": tp,
        #             "sensitivity": sensitivity,
        #             "specificity": specificity,
        #         }
        #     )

        return np.mean(val_score), {"Val loss": validation_loss}

    def save_model(self, path: str):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # def test(self, test_loader: DataLoader):
    #     """
    #     Method for performing the test of the model based on the given test loader.
    #
    #     The method computes precision, recall, F1-score, and other useful classification metrics.
    #
    #     :param test_loader: data loader for test data
    #     :return: a dictionary containing the value of each metric average across the test examples
    #     """
    #     # create dictionary where the results have to be stored
    #     results = {}
    #     # prepare predictions and targets for evaluation
    #     # TODO prepare_for_evaluation was changed to return three values instead of two, rest of the function requires fixing
    #     users, pos_preds, neg_preds = self.prepare_for_evaluation(test_loader)
    #     # compute metrics
    #     results["fbeta-1.0"] = compute_metric("fbeta-1.0", preds, targets)
    #     p, r, f, _ = precision_recall_fscore_support(
    #         targets, preds, beta=1.0, average=None
    #     )
    #     results["neg_prec"] = p[0]
    #     results["pos_prec"] = p[1]
    #     results["neg_rec"] = r[0]
    #     results["pos_rec"] = r[1]
    #     results["neg_f"] = f[0]
    #     results["pos_f"] = f[1]
    #     results["tn"], results["fp"], results["fn"], results["tp"] = (
    #         int(i) for i in tuple(confusion_matrix(targets, preds).ravel())
    #     )
    #     results["sensitivity"] = results["tp"] / (results["tp"] + results["fn"])
    #     results["specificity"] = results["tn"] / (results["tn"] + results["fp"])
    #     results["acc"] = accuracy_score(targets, preds)
    #
    #     return results
