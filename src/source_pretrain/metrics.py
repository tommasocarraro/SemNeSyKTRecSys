from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, fbeta_score

ranking_metrics = ["ndcg", "hit"]
Valid_Metrics_Type = Literal["AUC", "NDCG", "NDCG@10", "HR", "HR@10"]


def auc(users: NDArray, pos_preds: NDArray, neg_preds: NDArray) -> float:
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


def ndcg_at_k(pred_scores: NDArray, ground_truth: NDArray, k=10) -> float:
    """
    Computes the NDCG (at k) given the predicted scores and relevance of the items.

    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: NDCG at k position
    """
    k = min(pred_scores.shape[1], k)
    # compute DCG
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    log_term = 1. / np.log2(np.arange(2, k + 2))
    # compute metric
    dcg = (rank_relevance * log_term).sum(axis=1)
    # compute IDCG
    # idcg is the ideal ranking, so all the relevant items must be at the top, namely all 1 have to be at the top
    idcg = np.array([(log_term[:min(int(n_pos), k)]).sum() for n_pos in ground_truth.sum(axis=1)])
    return dcg / idcg


def hit_at_k(pred_scores: NDArray, ground_truth: NDArray, k=10) -> bool:
    """
    Computes the hit ratio (at k) given the predicted scores and relevance of the items.

    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: hit ratio at k position
    """
    k = min(pred_scores.shape[1], k)
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    # sum along axis 1 to count number of relevant items on first k-th positions
    # it is enough to have one relevant item in the first k-th for having a hit ratio of 1
    return rank_relevance.sum(axis=1) > 0


def compute_metric(
    metric: Valid_Metrics_Type,
    preds: Union[NDArray, tuple[NDArray, NDArray]],
    ground_truth: Optional[NDArray] = None,
    users: Optional[NDArray] = None,
) -> Union[float, bool]:
    """
    Compute the given metric on the given predictions and ground truth.

    :param metric: name of the metric that has to be computed
    :param preds: either the predicted scores or the positive&negative predictions
    :param ground_truth: target ratings for validation user-item pairs
    :param users: user indexes. Optional. At the moment, only used on AUC computation.
    :return: the value of the given metric for the given predictions and ground truth
    """
    if isinstance(preds, tuple):
        pos_preds, neg_preds = preds
        if metric.startswith("AUC"):
            return auc(users=users, pos_preds=pos_preds, neg_preds=neg_preds)
    else:
        kws = {"k": int(metric.split("@")[1])} if "@" in metric else {}
        if metric.startswith("NDCG"):
            return ndcg_at_k(pred_scores=preds, ground_truth=ground_truth, **kws)
        elif metric.startswith("HR"):
            return hit_at_k(pred_scores=preds, ground_truth=ground_truth, **kws)

    raise ValueError("Invalid parameters combination")
