from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, fbeta_score

valid_metrics = ["mse", "rmse", "fbeta", "acc", "auc"]


def mse(pred_scores, ground_truth):
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


def fbeta(pred_scores, ground_truth, beta):
    """
    Computes the f-beta measure between predictions and targets with the given beta value.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: f-beta measure
    """
    return fbeta_score(
        ground_truth, pred_scores, beta=beta, pos_label=0, average="binary"
    )


def acc(pred_scores, ground_truth):
    """
    Computes the accuracy between predictions and targets.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: accuracy
    """
    return accuracy_score(ground_truth, pred_scores)


def isfloat(num: str):
    """
    Check if a string contains a float.

    :param num: string to be checked
    :return: True if num is float, False otherwise
    """
    try:
        float(num)
        return True
    except ValueError:
        return False


def check_metrics(metrics: Union[str, list[str]]):
    """
    Check if the given list of metrics' names is correct.

    :param metrics: list of str containing the name of some metrics
    """
    err_msg = f"Some of the given metrics are not valid. The accepted metrics are {valid_metrics}"
    if isinstance(metrics, str):
        metrics = [metrics]
    assert all(
        [isinstance(m, str) for m in metrics]
    ), "The metrics must be represented as strings"
    assert all([m in valid_metrics for m in metrics if "-" not in m]), err_msg
    assert all(
        [
            m.split("-")[0] in valid_metrics and isfloat(m.split("-")[1])
            for m in metrics
            if "-" in m
        ]
    ), err_msg


def auc(pos_preds: NDArray, neg_preds: NDArray):
    # TODO the average here is general, is it maybe better to do it at the user level and then average again?
    """
    Computes the AUC score between predicted and target ratings. The given scores are aligned, namely the first positive
    score has to be compared with the first negative score. In other words, they correspond to the same user.

    :param pos_preds: scores for positive interactions
    :param neg_preds: scores for negative interactions
    :return: average AUC score across all the validation set
    """
    return np.mean((pos_preds - neg_preds) > 0)


def compute_metric(metric: str, pred_scores: NDArray, ground_truth: NDArray = None):
    """
    Compute the given metric on the given predictions and ground truth.

    :param metric: name of the metric that has to be computed
    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
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
            return auc(pred_scores, ground_truth)
