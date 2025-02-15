from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from tqdm import tqdm


def get_reg_axiom_data(
    src_ui_matrix: csr_matrix,
    tgt_ui_matrix: csr_matrix,
    tgt_sparsity: float,
    n_sh_users: int,
    sim_matrix: csr_matrix,
    top_k_items: NDArray,
    save_dir_path: Path,
    src_dataset_name: str,
    tgt_dataset_name: str,
) -> dict[int, NDArray]:
    """
    This function generates user-item pairs that will be given in input to the second axiom of the LTN model, namely the
    axiom that perform logical regularization based on information transferred from the source domain to the target
    domain.

    :param src_ui_matrix: sparse user-item matrix containing positive interactions in the source domain. It is used to
    determine the cold-start users in the source domain. Knowledge cannot be transferred from these users as the model
    learned little information.
    :param tgt_ui_matrix: sparse user-item matrix containing positive interactions in the target domain
    :param tgt_sparsity: sparsity factor of the target domain
    :param n_sh_users: number of shared users across domains
    :param sim_matrix: similarity matrix containing a 1 if there exists a path between the source and target item,
    0 otherwise
    :param top_k_items: numpy array containing for each shared user a list of top-k items the user might like, generated
    using a recommendation model pre-trained in the source domain
    :param save_dir_path: path to save the generated reg axiom data
    :param src_dataset_name: name of the source domain dataset to use
    :param tgt_dataset_name: name of the target domain dataset to use
    """
    save_dir_file_path = (
        save_dir_path / f"{src_dataset_name}_{tgt_dataset_name}_tgt_sparsity={tgt_sparsity}_reg_axiom.h5"
    )
    if save_dir_file_path.is_file():
        logger.debug(f"Found precomputed user-item pairs at {save_dir_file_path}. Loading it...")
        return load_from_hdf5(save_dir_file_path)

    processed_interactions = defaultdict(set)
    # find IDs of source domain items for which there exists a path with at least one target domain item
    src_exist_path_items = set(sim_matrix.nonzero()[0])
    # find IDs of target domain items for which there exists a path with at least one source domain item
    tgt_exist_path_items = set(sim_matrix.nonzero()[1])
    tgt_exist_path_items_l = list(tgt_exist_path_items)
    # iterate through each user in the set of shared users (the shared users are the only ones for which the information
    # can be transferred)
    for user in tqdm(range(n_sh_users), desc="Generating user-item pairs as input to LTN model", dynamic_ncols=True):
        # check if the user is cold-start in the source domain
        user_src_interacted_items = src_ui_matrix[user].nonzero()[1]
        if len(user_src_interacted_items) > 5:
            # take the top k items recommended for this user in the source domain (assuming k to be low and the
            # recommender to be accurate, these should be items for which we have very accurate predictions in the
            # source domain)
            user_src_top_k = top_k_items[user]
            # check which of these items are connected to at least one item in the target domain. Note also that the
            # resulting items will be items with more than 200 ratings in the source domain (warm start items) cause the
            # sim matrix only contains paths for items with more than 200 ratings in the source domain
            filtered_user_src_top_k = [j for j in user_src_top_k if j in src_exist_path_items]
            # check if this list is not empty, namely if there exists at least a source domain item that this user likes
            # and is connected to at least one target domain item
            # if this does not exist, then the user should not be considered as any paths will be included in the
            # sim matrix for this user
            if filtered_user_src_top_k:
                # take all the items that have been positively interacted by the shared user in the target domain
                user_tgt_interacted_items = tgt_ui_matrix[user].nonzero()[1]
                # check if the user is a cold-start user. We want to transfer information only to cold-start users in
                # target domain as the model should have enough data for the other users to learn enough information
                if len(user_tgt_interacted_items) <= 5:
                    # get all the items for which a rating is missing in the target domain for this cold-start user
                    all_items = np.arange(tgt_ui_matrix.shape[1])
                    # for each item, check if the item is connected with at least one source domain item
                    no_rated_items = np.setdiff1d(all_items, user_tgt_interacted_items)

                    # pre-compute paths existence for all top-k items for the shared user
                    user_src_tgt_paths = sim_matrix[filtered_user_src_top_k]

                    # get indices of target domain items for which a path to the source domain item exists each set in
                    # this list contains paths from a source domain item to all the connected target domain items
                    user_paths = [set(user_paths.nonzero()[1]) for user_paths in user_src_tgt_paths]

                    for tgt_item_id in np.intersect1d(no_rated_items, tgt_exist_path_items_l):
                        # if the connection exists, we want to find to which source domain items liked by the shared
                        # user this item is connected

                        # for each possible path from each of the top-k source items for the shared user, we check
                        # if the path converges into the current target item
                        for src_item_id, path_indices in zip(filtered_user_src_top_k, user_paths):
                            if tgt_item_id in path_indices and tgt_item_id not in processed_interactions[user]:
                                # if the current path converges into the current target item, add it to the set
                                processed_interactions[user].add(tgt_item_id)

    logger.debug("Saving the reg axiom data to file system")
    save_to_hdf5(processed_interactions, save_dir_file_path)

    return {k: np.array(list(processed_interactions[k])) for k in processed_interactions.keys()}


def save_to_hdf5(data: dict[int, set[int]], save_file_path: Path):
    """
    Save the given dictionary to an HDF5 file

    :param data: dictionary to save to file system
    :param save_file_path: path to save the data to
    """
    with h5py.File(save_file_path, "w") as f:
        for key, values in data.items():
            f.create_dataset(str(key), data=np.array(list(values), dtype=np.int32), dtype=np.int32)


def load_from_hdf5(save_file_path: Path) -> dict[int, NDArray]:
    """
    Load the given HDF5 file

    :param save_file_path: path where the HDF5 file is stored
    :return: a dictionary containing the loaded data
    """
    data = {}
    with h5py.File(save_file_path, "r") as f:
        for key in f.keys():
            data[int(key)] = f[key][:]
    return data


def sample_neg_items(
    sampled_sh_users: list[int], processed_interactions: dict[int, NDArray], tgt_ui_matrix: csr_matrix
) -> list[NDArray]:
    """
    Sample one negative item for each user in sampled_sh_users

    :param sampled_sh_users: sampled shared users
    :param processed_interactions: user-item interactions for which the sampling for the regularization axiom has to be
    performed
    :param tgt_ui_matrix: target domain user-item interactions
    :return: one negative item for each user in the sampled shared users
    """
    # extract for each shared user the items in the target domain which may be candidates for positive items
    sh_users_maybe_pos_items = {u: processed_interactions[u] for u in sampled_sh_users if u in processed_interactions}
    # extract for each shared user the items which received positive ratings
    sh_users_pos_items = [tgt_ui_matrix[user].indices for user in sampled_sh_users]

    negative_samples = []
    # compute the range of all item ids in the dataset
    all_items = np.arange(tgt_ui_matrix.shape[1])
    for user_index, user_id in enumerate(sampled_sh_users):
        # get the positive items candidates for the given user, default to empty list if there are none
        maybe_pos = sh_users_maybe_pos_items.get(user_id, [])
        # perform set difference between the whole range of items and the union of positive items and positive candidates
        negative_candidates = np.setdiff1d(all_items, np.union1d(maybe_pos, sh_users_pos_items[user_index]))
        # sample one negative from the negative candidates
        sampled_negatives = np.random.choice(negative_candidates, replace=False)
        negative_samples.append(sampled_negatives)

    return negative_samples
