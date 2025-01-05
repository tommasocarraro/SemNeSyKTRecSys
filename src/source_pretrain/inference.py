import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.nn.functional import softmax
from tqdm import tqdm

from src.device import device


def generate_pre_trained_src_matrix(
    mf_model, best_weights, n_shared_users, k, batch_size
) -> csr_matrix:
    """
    This function takes the pre-trained MF model in the source domain and generates a ranking of source domain items for
    each shared user. The first k position in the ranking are the top recommended items
    that are then used by the LTN model to transfer knowledge.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param n_shared_users: number of shared users across domains
    :param k: number of items at the top of the ranking that have to be treated as positive items.
    :param batch_size: number of predictions to be computed in parallel at each prediction step.
    """
    # load the best weights on the model
    mf_model.load_state_dict(torch.load(best_weights, map_location=device))
    # initialize predictions tensor
    preds = torch.zeros((n_shared_users, mf_model.n_items))
    # compute predictions for all shared users and items pairs and put them in the preds tensor
    # each row is a user and contains the predictions for all the items in the catalog for that user
    for u in tqdm(range(n_shared_users)):
        for start_idx in range(0, mf_model.n_items, batch_size):
            end_idx = min(start_idx + batch_size, mf_model.n_items)
            users = torch.full((end_idx - start_idx,), u, dtype=torch.long, device=device)
            items = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                preds[u, start_idx:end_idx] = mf_model(users, items)

    # create the rankings for each user and take the indexes of the items in the first k positions
    # these will be the items for which the model is more certain and that will be used for transferring knowledge to
    # the target domain
    pos_items = torch.argsort(preds, dim=1, descending=True)[:, :k].numpy()
    return pos_items
