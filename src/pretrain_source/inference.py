from pathlib import Path

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from src.device import device
from src.model import MatrixFactorization


def generate_pre_trained_src_matrix(
    mf_model: MatrixFactorization,
    best_weights_path: Path,
    n_shared_users: int,
    batch_size: int,
    save_dir_path: Path,
    upper_bound=200,
) -> Tensor:
    """
    This function takes the pre-trained MF model in the source domain and generates a ranking of source domain items for
    each shared user. The first k position in the ranking are the top recommended items
    that are then used by the LTN model to transfer knowledge.

    :param mf_model: architecture of the Matrix Factorization model whose best weights have to be loaded
    :param best_weights_path: path to the best weights that have to be loaded in the Matrix Factorization architecture
    :param n_shared_users: number of shared users across domains
    :param batch_size: number of predictions to be computed in parallel at each prediction step.
    :param save_dir_path: path to save the generated rankings
    :param upper_bound: how many top predictions per user to return
    :return: a tensor containing the top upper_bound predictions per user
    """
    model_name = best_weights_path.stem
    save_file_path = save_dir_path / f"{model_name}_top_200_preds.npy"

    if save_file_path.is_file():
        return torch.load(save_file_path, map_location=device, weights_only=True)

    # put model on correct device
    mf_model = mf_model.to(device)
    # load the best weights on the model
    mf_model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    logger.debug(f"Loaded source model's weights from {best_weights_path}")
    # create an empty tensor on CPU to avoid using too much VRAM
    preds = torch.empty((n_shared_users, mf_model.n_items), device="cpu")
    # compute predictions for all shared users and items pairs and put them in the preds tensor
    # each row is a user and contains the predictions for all the items in the catalog for that user
    for u in tqdm(
        range(n_shared_users), desc="Generating dense interactions matrix for the source domain", dynamic_ncols=True
    ):
        for start_idx in range(0, mf_model.n_items, batch_size):
            end_idx = min(start_idx + batch_size, mf_model.n_items)
            users = torch.full((end_idx - start_idx,), u, dtype=torch.int32, device=device)
            items = torch.arange(start_idx, end_idx, dtype=torch.int32, device=device)
            with torch.no_grad():
                preds[u, start_idx:end_idx] = mf_model(users, items)

    # create the rankings for each user and take the indexes of the items in the first k positions these will be the
    # items for which the model is more certain and that will be used for transferring knowledge to the target domain
    logger.debug(
        f"Finished generating dense interactions matrix. Sorting and collecting the top {upper_bound} predictions"
    )
    pos_items = torch.argsort(preds, dim=1, descending=True)[:, :upper_bound]

    logger.debug("Storing the dense interactions matrix in the file system")
    torch.save(pos_items, save_file_path)

    return pos_items
