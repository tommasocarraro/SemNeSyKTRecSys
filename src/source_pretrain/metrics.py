from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, fbeta_score

Valid_Metrics_Type = Literal["mse", "rmse", "fbeta", "acc", "auc"]


def mse(pred_scores: NDArray, ground_truth: NDArray):
    """
    Computes the Squared Error between predicted and target ratings.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: Squared error for each user
    """
    assert (
        pred_scores.shape == ground_truth.shape
    ), "predictions and targets must match in shape."
    return np.square(pred_scores - ground_truth)


def fbeta(pred_scores: NDArray, ground_truth: NDArray, beta: float):
    """
    Computes the f-beta measure between predictions and targets with the given beta value.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :param beta: ratio of recall importance to precision importance
    :return: f-beta measure
    """
    return fbeta_score(
        ground_truth, pred_scores, beta=beta, pos_label=0, average="binary"
    )


def acc(pred_scores: NDArray, ground_truth: NDArray):
    """
    Computes the accuracy between predictions and targets.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: accuracy
    """
    return accuracy_score(ground_truth, pred_scores)


def auc(users: NDArray, pos_preds: NDArray, neg_preds: NDArray):
    """
    Computes the AUC score between predicted and target ratings. The given scores are aligned, namely the first positive
    score has to be compared with the first negative score. In other words, they correspond to the same user.

    :param users: user indexes for user-level mean
    :param pos_preds: scores for positive interactions
    :param neg_preds: scores for negative interactions
    :return: average AUC score across all the validation set
    """
    single_values = (pos_preds - neg_preds) > 0
    # get unique users and their indices
    _, inverse_indices = np.unique(users, return_inverse=True)
    # group single_values by user
    user_sums = np.bincount(inverse_indices, weights=single_values)
    user_counts = np.bincount(inverse_indices)
    # compute user-level means
    user_means = user_sums / user_counts
    # final mean across all users
    final_mean_auc = user_means.mean()
    return final_mean_auc


def compute_metric(
    metric: Valid_Metrics_Type,
    pred_scores: NDArray,
    ground_truth: Optional[NDArray] = None,
    users: Optional[NDArray] = None,
):
    """
    Compute the given metric on the given predictions and ground truth.

    :param metric: name of the metric that has to be computed
    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :param users: user indexes. Optional. At the moment, only used on AUC computation.
    :return: the value of the given metric for the given predictions and ground truth
    """
    if "-" in metric:
        m, beta = metric.split("-")
        beta = float(beta)
        return fbeta(pred_scores, ground_truth, beta)
    else:
        if metric == "mse" or metric == "rmse":
            return mse(pred_scores, ground_truth)
        elif metric == "acc":
            return acc(pred_scores, ground_truth)
        elif metric == "auc":
            return auc(users, pred_scores, ground_truth)
