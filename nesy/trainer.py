import torch
import wandb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from nesy.metrics import compute_metric, check_metrics


class Trainer:
    """
    Abstract base class that manages the training of the model.

    Each model implementation must inherit from this class and implement the train_epoch() method. It is also possible
    to use overloading to redefine the behavior of some methods.
    """
    def __init__(self, model, optimizer, wandb_train=False):
        """
        Constructor of the trainer.

        :param model: neural model for which the training has to be performed
        :param optimizer: optimizer that has to be used for the training of the model
        :param wandb_train: whether to log data on Weights and Biases servers. This is used when using hyper-parameter
        optimization
        """
        self.model = model
        self.optimizer = optimizer
        self.wandb_train = wandb_train

    def train(self, train_loader, val_loader, val_metric=None, n_epochs=500, early=None, verbose=10, save_path=None):
        """
        Method for the train of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 500
        :param early: patience for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details
        :param save_path: path where to save the best model, default to None
        """
        if val_metric is not None:
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
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = "Epoch %d - Train loss %.3f - Val %s %.3f" % (epoch + 1, train_loss, val_metric, val_score)
                # add log_dict to log_record
                log_record += " - " + " - ".join(["%s %.3f" % (k, v) for k, v in log_dict.items() if k != "train_loss"])
                # print epoch report
                print(log_record)
                if self.wandb_train:
                    # log validation metric value
                    wandb.log({"smooth_%s" % (val_metric,): val_score})
                    # log training information
                    wandb.log(log_dict)
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

    def train_epoch(self, train_loader, epoch=None):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :param epoch: index of epoch
        :return: training loss value averaged across training batches and a dictionary containing useful information
        to log, such as other metrics computed by this model
        """
        raise NotImplementedError()

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            return self.model(u_idx, i_idx, dim)

    def prepare_for_evaluation(self, loader):
        """
        It prepares an array of predictions and targets for computing classification metrics.

        :param loader: loader containing evaluation data
        :return: predictions and targets
        """
        preds, targets = [], []
        for batch_idx, (X, y) in enumerate(loader):
            preds_ = self.predict(X)
            preds.append(preds_)
            targets.append(y.numpy())
        return np.concatenate(preds), np.concatenate(targets)

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name (it is F0.5-score for source domain and F1-score for target domain)
        :return: validation score based on the given metric averaged across all validation examples
        """
        # prepare predictions and targets for evaluation
        preds, targets = self.prepare_for_evaluation(val_loader)
        # compute F-score
        val_score = compute_metric(val_metric, preds, targets)
        # compute precision and recall
        p, r, f, _ = precision_recall_fscore_support(targets, preds,
                                                     beta=float(val_metric.split("-")[1])
                                                     if "fbeta" in val_metric else 1.0, average=None)
        # compute other useful metrics used in classification tasks
        tn, fp, fn, tp = tuple(confusion_matrix(targets, preds).ravel())
        sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        # log metrics to WandB servers
        if self.wandb_train:
            wandb.log({
                "neg_prec": p[0],
                "pos_prec": p[1],
                "neg_rec": r[0],
                "pos_rec": r[1],
                "neg_f": f[0],
                "pos_f": f[1],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "sensitivity": sensitivity,
                "specificity": specificity
            })

        return val_score

    def save_model(self, path):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, test_loader):
        """
        Method for performing the test of the model based on the given test loader.

        The method computes precision, recall, F1-score, and other useful classification metrics.

        :param test_loader: data loader for test data
        :return: a dictionary containing the value of each metric average across the test examples
        """
        # create dictionary where the results have to be stored
        results = {}
        # prepare predictions and targets for evaluation
        preds, targets = self.prepare_for_evaluation(test_loader)
        # compute metrics
        results["fbeta-1.0"] = compute_metric("fbeta-1.0", preds, targets)
        p, r, f, _ = precision_recall_fscore_support(targets, preds, beta=1., average=None)
        results["neg_prec"] = p[0]
        results["pos_prec"] = p[1]
        results["neg_rec"] = r[0]
        results["pos_rec"] = r[1]
        results["neg_f"] = f[0]
        results["pos_f"] = f[1]
        results["tn"], results["fp"], results["fn"], results["tp"] = \
            (int(i) for i in tuple(confusion_matrix(targets, preds).ravel()))
        results["sensitivity"] = results["tp"] / (results["tp"] + results["fn"])
        results["specificity"] = results["tn"] / (results["tn"] + results["fp"])
        results["acc"] = accuracy_score(targets, preds)

        return results
