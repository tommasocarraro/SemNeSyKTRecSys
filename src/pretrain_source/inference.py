from pathlib import Path

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch import Tensor
from tqdm import tqdm

from src.device import device
from src.model import MatrixFactorization


def generate_pre_trained_src_matrix(
    mf_model: MatrixFactorization,
    best_weights_path: Path,
    src_ui_matrix: csr_matrix,
    n_shared_users: int,
    save_dir_path: Path,
) -> NDArray:
    """
    This function takes the pre-trained MF model in the source domain and generates a ranking of source domain items for
    each shared user. The first k position in the ranking are the top recommended items
    that are then used by the LTN model to transfer knowledge.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights_path: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param src_ui_matrix: user-item ratings matrix from the source domain
    :param n_shared_users: number of shared users across domains
    :param save_dir_path: path to save the generated rankings
    :return: a tensor containing the top upper_bound predictions per user
    """
    model_name = best_weights_path.stem
    save_file_path = save_dir_path / f"{model_name}_top_200_preds.npy"

    if save_file_path.is_file():
        logger.debug(f"Found precomputed interactions matrix for the source domain at {save_file_path}. Loading it..")
        return np.load(save_file_path, allow_pickle=True)

    # put model on correct device
    mf_model = mf_model.to(device)
    # load the best weights on the model
    mf_model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    logger.debug(f"Loaded source model's weights from {best_weights_path}")

    # compute the top rankings for each user and save the indexes of the items appearing in them, which represent the
    # items for which the model is more confident and that will be used for transferring knowledge to the target domain
    top_preds_all_users = []
    for u in tqdm(
        range(n_shared_users), desc="Generating dense interactions matrix for the source domain", dynamic_ncols=True
    ):
        # predicting on a single user, so simply replicate the user index for n_items times
        users = torch.full((mf_model.n_items,), u, dtype=torch.int32, device=device)
        # create a tensor containing the whole range of items in the dataset
        items = torch.arange(mf_model.n_items, dtype=torch.int32, device=device)
        # compute the ranking predictions through the model
        with torch.no_grad():
            preds: Tensor = mf_model(users, items)
        # obtain the sparse row of ratings for the user
        sparse_ratings = src_ui_matrix[u]
        # compute how many positive interactions the user provided
        n_ratings = sparse_ratings.count_nonzero()
        # set the range of values for k within [n_ratings, n_ratings * 5], which is a heuristic based on the fact that
        # we want at least n_ratings results for the user and at most a sizeable amount of rankings proportional to n_ratings
        min_k = n_ratings
        max_k = n_ratings * 5
        k_range = np.arange(min_k, max_k + 1)
        # obtain the whole list of ratings provided by the user
        ground_truth = sparse_ratings.toarray()
        # compute the value of k within the given range such that it maximizes NDCG@k for the user
        best_k = compute_best_k(
            ground_truth=ground_truth, preds=np.asarray([preds.detach().cpu().numpy()]), k_range=k_range
        )
        # compute the top k rankings for the user
        top_preds = torch.topk(preds, best_k).indices
        top_preds_all_users.append(top_preds.detach().cpu().numpy())

    logger.debug(f"Finished generating dense interactions matrix")
    pos_items = np.array(top_preds_all_users, dtype=object)

    logger.debug("Storing the dense interactions matrix in the file system")
    np.save(arr=pos_items, file=save_file_path)

    return pos_items


def compute_best_k(ground_truth: NDArray, preds: NDArray, k_range: NDArray) -> int:
    """
    Find the value of k which maximizes the NDCG@k for a single user

    :param ground_truth: ground truth
    :param preds: predicted scores
    :param k_range: range of possible values for k
    :return: best k for a given user
    """
    n_items = ground_truth.shape[1]

    # sort the predictions in descending order and get the reordered preds
    sorted_indices = np.argsort(-preds, axis=1)
    preds_sorted = np.take_along_axis(ground_truth, sorted_indices, axis=1)

    # calculate the position discounts
    position_indices = np.arange(2, n_items + 2)
    discounts = 1.0 / np.log2(position_indices)

    # calculate the DCG
    dcg = np.cumsum(preds_sorted * discounts, axis=1)

    # calculate the IDCG
    ideal_sorted = -np.sort(-ground_truth, axis=1)
    idcg = np.cumsum(ideal_sorted * discounts, axis=1)

    # select the DCG and IDCG values for the values of k within the range, while subtracting 1 from them to account
    # for off-by-one
    dcg_at_k = dcg[:, k_range - 1]
    idcg_at_k = idcg[:, k_range - 1]

    # in case idcg_at_k is zero, replace it with epsilon to avoid runtime errors
    idcg_at_k = np.maximum(idcg_at_k, np.finfo(float).eps)

    # calculate the NDCG
    ndcg_scores = dcg_at_k / idcg_at_k
    # get the index of the value of k which maximizes the NDCG
    best_idx = np.argmax(ndcg_scores)
    best_k = int(k_range[best_idx])

    return best_k
