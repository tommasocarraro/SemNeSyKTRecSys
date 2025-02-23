from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import torch
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch import Tensor
from tqdm import tqdm

from src.device import device


def get_reg_axiom_data(
    src_ui_matrix: csr_matrix,
    tgt_ui_matrix: csr_matrix,
    sparsity_sh: Optional[float],
    n_sh_users: int,
    sim_matrix: csr_matrix,
    top_k_items: NDArray,
    save_dir_path: Path,
    src_dataset_name: str,
    tgt_dataset_name: str,
    retrained_model: bool,
) -> dict[int, Tensor]:
    """
    This function generates user-item pairs that will be given in input to the second axiom of the LTN model, namely the
    axiom that perform logical regularization based on information transferred from the source domain to the target
    domain.

    :param src_ui_matrix: sparse user-item matrix containing positive interactions in the source domain. It is used to
    determine the cold-start users in the source domain. Knowledge cannot be transferred from these users as the model
    learned little information.
    :param tgt_ui_matrix: sparse user-item matrix containing positive interactions in the target domain
    :param sparsity_sh: TODO
    :param n_sh_users: number of shared users across domains
    :param sim_matrix: similarity matrix containing a 1 if there exists a path between the source and target item,
    0 otherwise
    :param top_k_items: numpy array containing for each shared user a list of top-k items the user might like, generated
    using a recommendation model pre-trained in the source domain
    :param save_dir_path: path to save the generated reg axiom data
    :param src_dataset_name: name of the source domain dataset to use
    :param tgt_dataset_name: name of the target domain dataset to use
    """
    if sparsity_sh is not None:
        tgt_n_ratings_sh_str = f"_sparsity_sh={sparsity_sh}"
    else:
        tgt_n_ratings_sh_str = ""

    save_dir_file_path = save_dir_path / f"{src_dataset_name}_{tgt_dataset_name}{tgt_n_ratings_sh_str}_reg_axiom.h5"
    if save_dir_file_path.is_file() and not retrained_model:
        logger.debug(f"Found precomputed user-item pairs at {save_dir_file_path}. Loading it...")
        return load_from_hdf5(save_dir_file_path)

    # create the set of all user IDs shared between the two domains
    sh_users = set(range(n_sh_users))

    # compute the mappings from source domain items to target domain items through the similarity matrix
    src_to_tgt_sim: dict[int, set[int]] = defaultdict(set)
    for src_item, tgt_item in zip(*sim_matrix.nonzero()):
        src_to_tgt_sim[src_item].add(tgt_item)

    # compute the mapping from source domain shared users to their ratings
    user_to_ratings_src: dict[int, set[int]] = defaultdict(set)
    for user, item in zip(*src_ui_matrix.nonzero()):
        if user in sh_users:
            user_to_ratings_src[user].add(item)

    # compute the set of warm shared users (more than 5 ratings in the source domain)
    warm_users_src = {user for user, ratings in user_to_ratings_src.items() if len(ratings) > 5}
    del user_to_ratings_src

    # compute the mapping from target domain shared users to their ratings
    user_to_ratings_tgt: dict[int, set[int]] = {user: set() for user in sh_users}
    for user, item in zip(*tgt_ui_matrix.nonzero()):
        if user in sh_users:
            user_to_ratings_tgt[user].add(item)

    # compute the set of cold shared users (less than 5 ratings in the target domain)
    cold_users_tgt = {user for user, ratings in user_to_ratings_tgt.items() if len(ratings) <= 5}

    # precompute set of all the target domain's item IDs
    all_item_ids = set(range(tgt_ui_matrix.shape[1]))

    # the transfer users are shared users who are cold in the target domain but warm in the source domain
    transfer_users = list(cold_users_tgt & warm_users_src)

    processed_interactions: dict[int, Tensor] = {}
    for user in tqdm(transfer_users, desc="Generating user-item pairs as input to LTN model", dynamic_ncols=True):
        # get ranking prediction for user
        user_top_k = top_k_items[user]
        # get target domain item IDs for items connected to the user's top k
        tgt_positive_candidates: set[int] = set()
        for src_item in user_top_k:
            if src_item in src_to_tgt_sim:
                tgt_positive_candidates.update(src_to_tgt_sim[src_item])
        # get the user's positive ratings in the target domain
        user_pos_ratings_tgt = user_to_ratings_tgt.get(user, set())
        # get the item IDs for which the user has given no rating
        user_no_ratings = all_item_ids - user_pos_ratings_tgt
        # for each user store the intersection between the unrated items and the positive candidates, if it's not empty
        negative_candidates = list(tgt_positive_candidates & user_no_ratings)
        if len(negative_candidates) > 0:
            processed_interactions[user] = torch.tensor(data=negative_candidates, dtype=torch.int32, device=device)

    logger.debug("Saving the reg axiom data to file system")
    save_to_hdf5(processed_interactions, save_dir_file_path)
    logger.debug("Reg axiom data saved to file system")

    return processed_interactions


def save_to_hdf5(data: dict[int, Tensor], save_file_path: Path):
    """
    Save the given dictionary to an HDF5 file

    :param data: dictionary to save to file system
    :param save_file_path: path to save the data to
    """
    with h5py.File(save_file_path, "w") as f:
        for key, values in data.items():
            f.create_dataset(name=str(key), data=values.detach().cpu().numpy())


def load_from_hdf5(save_file_path: Path) -> dict[int, Tensor]:
    """
    Load the given HDF5 file

    :param save_file_path: path where the HDF5 file is stored
    :return: a dictionary containing the loaded data
    """
    data = {}
    with h5py.File(save_file_path, "r") as f:
        for key in f.keys():
            data[int(key)] = torch.tensor(data=f[key][:], dtype=torch.int32, device=device)
    return data
