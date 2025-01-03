from typing import Generator

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def get_cold_ui(interactions: csr_matrix, n_sh_users: int):
    # Iterate through each row
    for i in tqdm(range(n_sh_users)):
        # Get the indices of the non-zero elements in the row
        non_zero_indices = interactions[i].nonzero()[1]

        # Check the number of non-zero elements
        if len(non_zero_indices) <= 5:
            # Efficiently find zero indices by excluding non-zero ones
            total_indices = np.arange(interactions.shape[1])
            zero_indices = np.setdiff1d(total_indices, non_zero_indices)

            for zer_idx in zero_indices:
                yield i, zer_idx


def filter_cold_ui(cold_ui: Generator, top_k_items: dict[int, list[int]], sim_matrix: csr_matrix):
    for user, item in tqdm(cold_ui):
        source_top_k = top_k_items[user]
        paths = sim_matrix[source_top_k]
        for path in tqdm(paths):
            if item in path.nonzero()[1]:
                yield user, path.nonzero()[1], item


# def get_cold_ui(interactions: csr_matrix):
#     # Initialize a list for storing results
#     cold_ui = []
#
#     # Iterate through each row
#     for i in tqdm(range(interactions.shape[0])):
#         # Get the indices of the non-zero elements in the row
#         non_zero_indices = interactions[i].nonzero()[1]
#
#         # Check the number of non-zero elements
#         if len(non_zero_indices) <= 5:
#             # Append the result to the list
#             cold_ui.append(np.column_stack((np.full(len(non_zero_indices), i), non_zero_indices)))
#
#     return np.concatenate(cold_ui, axis=0)
