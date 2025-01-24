from pathlib import Path

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from torch import Tensor
from tqdm import tqdm


def get_reg_axiom_data(
    src_ui_matrix: csr_matrix,
    tgt_ui_matrix: csr_matrix,
    tgt_sparsity: float,
    n_sh_users: int,
    sim_matrix: csr_matrix,
    top_k_items: Tensor,
    save_dir_path: Path,
    src_dataset_name: str,
    tgt_dataset_name: str,
):
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
        save_dir_path
        / f"{src_dataset_name}_{tgt_dataset_name}_k={top_k_items.shape[1]}_sparsity={tgt_sparsity}_reg_axiom.npy"
    )
    if save_dir_file_path.is_file():
        return np.load(save_dir_file_path)

    processed_interactions = []
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
            user_src_top_k = top_k_items[user].cpu().numpy()
            # check which of these items are connected to at least one item in the target domain. Note also that the
            # resulting items will be items with more than 300 ratings in the source domain (warm start items) cause the
            # sim matrix only contains paths for items with more than 300 ratings in the source domain
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
                            if tgt_item_id in path_indices:
                                # if the current path converges into the current target item, we generate the triplet
                                processed_interactions.append((user, src_item_id, tgt_item_id))

                                # if the length of interactions grows past a certain size
                                if len(processed_interactions) > 10000:
                                    # append to the file
                                    np.save(save_dir_file_path, np.array(processed_interactions))
                                    # and reset the list in order to avoid memory issues
                                    processed_interactions = []

    # save any remaining interactions
    if processed_interactions:
        np.save(save_dir_file_path, np.array(processed_interactions))
        del processed_interactions
    logger.debug(f"User-item pairs generated and stored in {save_dir_file_path}")

    logger.debug("Reloading the file and returning the dense array")
    return np.load(save_dir_file_path)
