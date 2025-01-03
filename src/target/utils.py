import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def get_reg_axiom_data(tgt_ui_matrix: csr_matrix,
                       n_sh_users: int,
                       sim_matrix: csr_matrix,
                       top_k_items: dict[int, list[int]]):
    """
    This function generates user-item pairs that will be given in input to the second axiom of the LTN model, namely the
    axiom that perform logical regularization based on information transferred from the source domain to the target
    domain.

    :param tgt_ui_matrix: sparse user-item matrix containing positive interactions in the target domain
    :param n_sh_users: number of shared users across domains
    :param sim_matrix: similarity matrix containing a 1 if there exists a path between the source and target item,
    0 otherwise
    :param top_k_items: dictionary containing for each shared users a list of top-k items the user might like, generated
    using a recommendation model pre-trained in the source domain
    """
    # find IDs of source domain items for which there exists a path with at least one target domain item
    src_exist_path_items = set(sim_matrix.nonzero()[0])
    # find IDs of target domain items for which there exists a path with at least one source domain item
    tgt_exist_path_items = set(sim_matrix.nonzero()[1])
    # iterate through each user in the set of shared users (the shared users are the only ones for which the information
    # can be transferred)
    for user in tqdm(range(n_sh_users)):
        # take the top k items recommended for this user in the source domain (assuming k to be low and the recommender
        # to be accurate, these should be items for which we have very accurate predictions in the source domain)
        user_src_top_k = top_k_items[user]
        # check which of these items are connected to at least one item in the target domain. Note also that the
        # resulting items will be items with more than 300 ratings in the source domain (warm start items) cause the
        # sim matrix only contains paths for items with more than 300 ratings in the source domain
        filtered_user_src_top_k = [j for j in user_src_top_k if j in src_exist_path_items]
        # check if this list is not empty, namely if there exists at least a source domain item that this user likes
        # and is connected to at least one target domain item
        # if this does not exist, then the user should not be considered as any paths will be included in the sim matrix
        # for this user
        if filtered_user_src_top_k:
            # take all the items that have been positively interacted by the shared user in the target domain
            non_zero_indices = tgt_ui_matrix[user].nonzero()[1]
            # check if the user is a cold-start user
            if len(non_zero_indices) <= 5:
                # get all the items for which a rating is missing in the target domain for this cold-start user
                total_indices = np.arange(tgt_ui_matrix.shape[1])
                zero_indices = np.setdiff1d(total_indices, non_zero_indices)

                for tgt_item_id in zero_indices:
                    # for each item, check if the item is connected with at least one source domain item
                    if tgt_item_id in tgt_exist_path_items:
                        # if the connection exists, we want to find to which source domain items liked by the shared
                        # user this item is connected

                        # get paths for the top-k items recommended for the shared user
                        user_src_tgt_paths = sim_matrix[filtered_user_src_top_k]
                        # for each possible path from each of the top-k source items for the shared user, we check if
                        # the path converges into the current target item
                        for src_item_id, user_paths in zip(filtered_user_src_top_k, user_src_tgt_paths):
                            if tgt_item_id in user_paths.nonzero()[1]:
                                # if the current path converges into the current target item, we generate the triplet
                                yield user, src_item_id, tgt_item_id
