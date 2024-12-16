import os
import random

import numpy as np
import torch
from tqdm import tqdm

from src import device


def set_seed(seed: int):
    """
    It sets the seed for the reproducibility of the experiments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def str_is_float(num: str):
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


def generate_pre_trained_src_matrix(
    mf_model, best_weights, pos_threshold, batch_size
) -> torch.Tensor:
    """
    Generates the dense source domain user-item matrix filled with predictions from a model pre-trained on the source
    domain. This will be the implementation of the LikesSource predicate in the LTN model.

    Since the model on the source domain is trained with a BPR criterion (ranking) and we need 0/1 values, we use the
    following heuristic to convert rankings into desired values:
    1. For each user, we create the ranking on the entire catalog of the source domain;
    2. We take the best pos_threshold positions, and we substitute them with a 1 in the returned matrix, 0 otherwise.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param pos_threshold: threshold to decide the number of best items in the ranking that have to be set to 1 in the
    final matrix.
    :param batch_size: number of predictions to be computed in parallel at each prediction step.
    """
    # load the best weights on the model
    mf_model.load_state_dict(
        torch.load(best_weights, map_location=device)["model_state_dict"]
    )
    preds = torch.zeros((mf_model.n_users, mf_model.n_items), device=device)
    for u in tqdm(range(mf_model.n_users)):
        for start_idx in range(0, mf_model.n_items, batch_size):
            end_idx = min(start_idx + batch_size, mf_model.n_items)
            users = torch.full(
                (end_idx - start_idx,), u, dtype=torch.long, device=device
            )
            items = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                preds[u, start_idx:end_idx] = mf_model(users, items)

    pos_idx = torch.argsort(preds, dim=1, descending=True)[:, :pos_threshold]

    final_ui_matrix = torch.zeros((mf_model.n_users, mf_model.n_items))
    user_idx = torch.arange(0, mf_model.n_users).repeat_interleave(pos_threshold).long()
    final_ui_matrix[user_idx, pos_idx.flatten()] = 1

    return final_ui_matrix.to(device)
