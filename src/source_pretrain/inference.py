import torch
from tqdm import tqdm
from src import device


def generate_pre_trained_src_matrix(
    mf_model, best_weights, n_shared_users, src_ui_matrix, pos_threshold, batch_size
) -> torch.Tensor:
    """
    Generates the dense source domain user-item matrix filled with predictions from a model pre-trained on the source
    domain. This will be the implementation of the LikesSource predicate in the LTN model.

    Note that the matrix will be n_shared_users x n_items because we can transfer knowledge just for shared users. The
    indexes from 0 to n_shared_users - 1 are the indexes of the shared users. So, in row 0, there will be predictions
    of the pre-trained model on the source domain for the user 0, that is shared. Note that in the original indexing
    the first n_shared_users indexes were dedicated to the shared users, so it is enough to compute predictions for
    this portion of users.

    Since the model on the source domain is trained with a BPR criterion (ranking) and we need 0/1 values, we use the
    following heuristic to convert rankings into desired values:
    1. For each user, we create the ranking on the entire catalog of the source domain;
    2. We take the best pos_threshold positions, and we substitute them with a 1 in the returned matrix, 0 otherwise.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param n_shared_users: number of shared users across domains
    :param src_ui_matrix: source domain user-item matrix
    :param pos_threshold: threshold to decide the number of best items in the ranking that have to be set to 1 in the
    final matrix.
    :param batch_size: number of predictions to be computed in parallel at each prediction step.
    """
    # load the best weights on the model
    mf_model.load_state_dict(
        torch.load(best_weights, map_location=device)["model_state_dict"]
    )
    # initialize predictions tensor
    preds = torch.zeros((n_shared_users, mf_model.n_items))
    # compute predictions for all shared users and items pairs and put them in the preds tensor
    for u in tqdm(range(n_shared_users)):
        for start_idx in range(0, mf_model.n_items, batch_size):
            end_idx = min(start_idx + batch_size, mf_model.n_items)
            users = torch.full(
                (end_idx - start_idx,), u, dtype=torch.long, device=device
            )
            items = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                preds[u, start_idx:end_idx] = mf_model(users, items)

    # create the rankings for each user and take the indexes of the items in the first pos_threshold positions
    pos_idx = torch.argsort(preds, dim=1, descending=True)[:, :pos_threshold]

    # create dense matrix with predictions where we put 1 if the item is in the first pos_threshold positions,
    # 0 otherwise
    # Convert to PyTorch sparse CSR tensor
    sh_users_src_ui_matrix = src_ui_matrix[:n_shared_users]
    crow_indices = torch.tensor(sh_users_src_ui_matrix.indptr, dtype=torch.int64)
    col_indices = torch.tensor(sh_users_src_ui_matrix.indices, dtype=torch.int64)
    values = torch.tensor(sh_users_src_ui_matrix.data, dtype=torch.float32)

    sh_users_src_ui_matrix = torch.sparse_csr_tensor(crow_indices, col_indices, values,
                                                     size=sh_users_src_ui_matrix.shape)

    user_idx = torch.arange(0, n_shared_users).repeat_interleave(pos_threshold).long()
    sh_users_src_ui_matrix[user_idx, pos_idx.flatten()] = 1

    return sh_users_src_ui_matrix.to(device)
