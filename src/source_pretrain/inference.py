import torch
from tqdm import tqdm
from src.device import device
from scipy.sparse import csr_matrix
import numpy as np
from torch.nn.functional import softmax


def generate_pre_trained_src_matrix(
    mf_model, best_weights, n_shared_users, src_ui_matrix, pos_threshold, batch_size
) -> csr_matrix:
    """
    Generates the dense source domain user-item matrix filled with predictions from a model pre-trained on the source
    domain. This will be the implementation of the LikesSource predicate in the LTN model.

    Note that the matrix will be n_shared_users x n_items because we can transfer knowledge just for shared users. The
    indexes from 0 to n_shared_users - 1 are the indexes of the shared users. So, in row 0, there will be predictions
    of the pre-trained model on the source domain for the user 0, that is the first shared user. Note that in the
    original indexing, the first n_shared_users indexes were dedicated to the shared users, so it is enough to compute
    predictions for this portion of users.

    Since the model on the source domain is trained with a BPR criterion (ranking) and we need 0/1 values, we use the
    following heuristic to convert rankings into desired values:
    1. For each user, we create the ranking on the entire catalog of the source domain;
    2. For each ranking, we take the first pos_threshold items and set them as positives in the final sparse matrix.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param n_shared_users: number of shared users across domains
    :param src_ui_matrix: source domain user-item matrix
    :param pos_threshold: threshold that determines from which ranking position items has to be predicted as negatives.
    All the items in the top pos_threshold positions will be classified as positives. All the items in ranking positions
    lower than pos_threshold will be classified as negatives.
    :param batch_size: number of predictions to be computed in parallel at each prediction step.
    """
    # TODO mettere cold start user tutto a zero sulla matrice LikesSource, cosi non trasferisce nulla
    # load the best weights on the model
    mf_model.load_state_dict(
        torch.load(best_weights, map_location=device)
    )
    # initialize predictions tensor
    preds = torch.zeros((1, mf_model.n_items))
    # compute predictions for all shared users and items pairs and put them in the preds tensor
    # each row is a user and contains the predictions for all the items in the catalog for that user
    for u in tqdm(range(n_shared_users)):
        for start_idx in range(0, mf_model.n_items, batch_size):
            end_idx = min(start_idx + batch_size, mf_model.n_items)
            users = torch.full(
                (end_idx - start_idx,), u, dtype=torch.long, device=device
            )
            items = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                preds[u, start_idx:end_idx] = mf_model(users, items)
        user_0_preds = preds[0]
        pass

    # create the rankings for each user and take the indexes of the items in the first pos_threshold positions
    # these will be the items that have to have a 1 in the LikesSource matrix
    pos_idx = torch.argsort(preds, dim=1, descending=True)[:, :pos_threshold].flatten().numpy()

    # create dense matrix with predictions where we put 1 if the item has a score that, once the softmax is applied,
    # is near 1, and 0 otherwise

    # take portion of user-item matrix dedicated to shared users
    sh_users_src_ui_matrix = src_ui_matrix[:n_shared_users]
    rows, cols = sh_users_src_ui_matrix.nonzero()
    ui_matrix_coords = set(zip(rows, cols))
    # compute the coordinates (rows, cols) of the new positions that have to be put at 1 (there are the predictions
    # of the pre-trained model)
    new_rows = np.arange(0, n_shared_users).repeat(pos_threshold)
    new_coords = set(zip(new_rows, pos_idx))
    # build final user-item matrix for the LikesSource predicate where there is a 1 if the pre-trained model predicted
    # the ui pair to be in a high rank or if there was a 1 originally in the dataset
    final_coords = new_coords | ui_matrix_coords
    rows, cols = zip(*final_coords)
    final_matrix = csr_matrix(
        ([1] * len(final_coords), (rows, cols)), sh_users_src_ui_matrix.shape
    )

    return final_matrix


def generate_pre_trained_src_matrix_2(
    mf_model, best_weights, n_shared_users, src_ui_matrix, beta, batch_size
) -> csr_matrix:
    """
    Generates the dense source domain user-item matrix filled with predictions from a model pre-trained on the source
    domain. This will be the implementation of the LikesSource predicate in the LTN model.

    Note that the matrix will be n_shared_users x n_items because we can transfer knowledge just for shared users. The
    indexes from 0 to n_shared_users - 1 are the indexes of the shared users. So, in row 0, there will be predictions
    of the pre-trained model on the source domain for the user 0, that is the first shared user. Note that in the
    original indexing, the first n_shared_users indexes were dedicated to the shared users, so it is enough to compute
    predictions for this portion of users.

    Since the model on the source domain is trained with a BPR criterion (ranking) and we need 0/1 values, we use the
    following heuristic to convert rankings into desired values:
    1. For each user, we create the ranking on the entire catalog of the source domain;
    2. We take the beta softmax of each ranking so that these are mapped to values near 0 and 1;
    3. Values near 0. will be converted to 0., while values near 1. will be converted to 1.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param n_shared_users: number of shared users across domains
    :param src_ui_matrix: source domain user-item matrix
    :param beta: hyper-parameter to be used for the softmax function that has to be applied to the ranking.
    :param batch_size: number of predictions to be computed in parallel at each prediction step.
    """
    # TODO mettere cold start user tutto a zero sulla matrice LikesSource, cosi non trasferisce nulla
    # load the best weights on the model
    mf_model.load_state_dict(
        torch.load(best_weights, map_location=device)
    )
    mf_model.n_items = 800
    # initialize predictions tensor
    preds = torch.zeros((n_shared_users, mf_model.n_items))
    # compute predictions for all shared users and items pairs and put them in the preds tensor
    # each row is a user and contains the predictions for all the items in the catalog for that user
    for u in tqdm(range(n_shared_users)):
        for start_idx in range(0, mf_model.n_items, batch_size):
            end_idx = min(start_idx + batch_size, mf_model.n_items)
            users = torch.full(
                (end_idx - start_idx,), u, dtype=torch.long, device=device
            )
            items = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                preds[u, start_idx:end_idx] = mf_model(users, items)

    # apply the softmax function with Beta to scale the predicted scores to 1/0
    scaled_scores = softmax(beta * preds, dim=1)

    # create dense matrix with predictions where we put 1 if the item has a score that, once the softmax is applied,
    # is near 1, and 0 otherwise

    # take portion of user-item matrix dedicated to shared users
    sh_users_src_ui_matrix = src_ui_matrix[:n_shared_users]
    rows, cols = sh_users_src_ui_matrix.nonzero()
    ui_matrix_coords = set(zip(rows, cols))
    # compute the coordinates (rows, cols) of the new positions that have to be put at 1 (there are the predictions
    # of the pre-trained model)
    # new_rows = np.arange(0, n_shared_users).repeat(pos_threshold)
    # new_coords = set(zip(new_rows, pos_idx))
    # # build final user-item matrix for the LikesSource predicate where there is a 1 if the pre-trained model predicted
    # # the ui pair to be in a high rank or if there was a 1 originally in the dataset
    # final_coords = new_coords | ui_matrix_coords
    # rows, cols = zip(*final_coords)
    # final_matrix = csr_matrix(
    #     ([1] * len(final_coords), (rows, cols)), sh_users_src_ui_matrix.shape
    # )
    #
    # return final_matrix
