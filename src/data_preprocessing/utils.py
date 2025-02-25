import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from orjsonl import orjsonl
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

from src.data_preprocessing.Dataset import Dataset
from src.data_preprocessing.Split_Strategy import SplitStrategy


def load_or_clear_dataset(save_file_path: Path, clear_saved_dataset: bool) -> Optional[Dataset]:
    """
    Either loads the already preprocessed dataset or reads the source and target datasets from the file system.

    :param save_file_path: file path pointing to the saved dataset
    :param clear_saved_dataset: whether to clear the saved dataset if it exists
    :return: the preprocessed dataset or the source and target datasets dataframes
    """
    # computing the path where to store the processed dataset at
    if save_file_path.is_file():
        if clear_saved_dataset:
            logger.debug("Clearing out previously computed dataset")
            os.remove(save_file_path)
        else:
            # if the dataset has already been processed before, simply load it with numpy
            logger.debug("Found precomputed dataset pair, reading it from file system")
            return np.load(save_file_path, allow_pickle=True).item()


def load_dataframes(first_dataset_path: Path, second_dataset_path: Path) -> tuple[DataFrame, DataFrame]:
    # get source and target ratings
    logger.debug("Reading the datasets' csv files with pandas")
    src_ratings = pd.read_csv(first_dataset_path, usecols=["userId", "itemId", "rating"])
    tgt_ratings = pd.read_csv(second_dataset_path, usecols=["userId", "itemId", "rating"])
    return src_ratings, tgt_ratings


def save_dataset(dataset: Dataset, save_file_path: Path) -> None:
    """
    Saves the dataset to the save_file_path

    :param dataset: dataset to save
    :param save_file_path: file path where the dataset should be saved
    """
    os.makedirs(save_file_path.parent, exist_ok=True)
    logger.info(f"Saving the dataset to file system")
    np.save(save_file_path, dataset)  # type: ignore


def create_ui_matrix(df: DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """
    Creates the user-item interaction matrix from the given dataframe.

    :param df: dataframe containing the ratings
    :param n_users: number of users
    :param n_items: number of items
    :return: a sparse matrix containing the user-item interactions
    """
    pos_ratings = df[df["rating"] == 1]
    return csr_matrix((pos_ratings["rating"], (pos_ratings["userId"], pos_ratings["itemId"])), shape=(n_users, n_items))


def increase_sparsity_sh(ratings: NDArray, sparsity_sh: float, sh_u_incr_ids: set[int], seed: int) -> NDArray:
    """
    Artificially increases the sparsity of the ratings dataframe by the given sparsity factor. I.e., if sparsity is 0.4,
    then 40% of each user's ratings will be retained in the dataset. The ratings are sampled randomly.

    :param ratings: numpy array containing the quadruples of the ratings
    :param sparsity_sh: the sparsity factor to use
    :param sh_u_incr_ids: the list of shared user ids
    :param seed: seed to use for sampling
    :return: the new ratings array and ui_matrix containing the sampled ratings
    """
    df = DataFrame(ratings, columns=["userId", "itemId", "rating"])

    if sparsity_sh == 0.0:
        df = df[~df["userId"].isin(sh_u_incr_ids)]
        return df.to_numpy()
    elif sparsity_sh == 1.0:
        return ratings

    desc = (
        f"Artificially increasing data sparsity for the shared users. "
        f"Selecting {sparsity_sh*100}% random ratings for each user"
    )
    tqdm.pandas(desc=desc, leave=False)

    def sample_ratings(x: DataFrameGroupBy):
        if x.name in sh_u_incr_ids:
            return x.sample(frac=sparsity_sh, random_state=seed)
        return x

    df = df.groupby("userId").progress_apply(sample_ratings).reset_index(drop=True)
    return df.to_numpy()


def remove_not_sh(ratings: NDArray, sh_u_ids: set[int]) -> NDArray:
    """
    Removes all non-shared user ratings from the given ratings array

    :param ratings: numpy array containing the triples of the ratings
    :param sh_u_ids: the set of shared user ids
    :return: numpy array containing the triples of the ratings of the shared users
    """
    df = DataFrame(ratings, columns=["userId", "itemId", "rating"])
    df = df[df["userId"].isin(sh_u_ids)]
    return df.to_numpy()


def make_save_file_path(
    save_dir_path: Path,
    src_dataset_path: Path,
    tgt_dataset_path: Path,
    tgt_n_ratings_sh: float,
    src_split_strategy: SplitStrategy,
    tgt_split_strategy: SplitStrategy,
    seed: int,
) -> Path:
    """
    Constructs the path where the dataset should be stored on file system by numpy

    :param save_dir_path: path of the directory where the dataset should be stored
    :param src_dataset_path: path of the raw csv dataset
    :param tgt_dataset_path: path to the target dataset
    :param tgt_n_ratings_sh: how many ratings to keep for the shared users on the target domain
    :param src_split_strategy: strategy used to split the source dataset
    :param tgt_split_strategy: strategy used to split the target dataset
    :param seed: random seed
    :return: Location of the processed dataset
    """
    os.makedirs(save_dir_path, exist_ok=True)
    source_file_name = (
        f"src_dataset={src_dataset_path.stem}_tgt_dataset={tgt_dataset_path.stem}tgt_n_ratings_sh={tgt_n_ratings_sh}_"
        f"{hash(src_split_strategy)}_{hash(tgt_split_strategy)}_seed={seed}.npy"
    )
    return save_dir_path / source_file_name


def reindex_users(
    ratings: DataFrame,
    sh_users_amzn_ids: set[str],
    sh_u_inc_ids: list[int],
    sh_users_amzn_id_to_incr_id: dict[str, int],
):
    """
    Re-indexes the users on the ratings dataframe in order to obtain incremental integer identifiers. Furthermore, the
    users shared between source and target domains are relocated to the first rows.

    :param ratings: the ratings dataframe
    :param sh_users_amzn_ids: the set of shared users
    :param sh_u_inc_ids: the list of shared user ids
    :param sh_users_amzn_id_to_incr_id: the mapping from user_id string to integer identifier
    """
    # get ids of non-shared users in source and target domains
    users = set(ratings["userId"]) - sh_users_amzn_ids
    u_ids = [(i + len(sh_u_inc_ids)) for i in range(len(users))]
    u_string_to_id = dict(zip(sorted(users), u_ids))
    u_string_to_id.update(sh_users_amzn_id_to_incr_id)
    ratings["userId"] = ratings["userId"].map(u_string_to_id)


def reindex_items(ratings: DataFrame) -> dict[str, int]:
    """
    Re-indexes the items on the ratings dataframe in order to obtain incremental integer identifiers.

    :param ratings: the ratings dataframe
    :return: the mapping from item id to integer identifier
    """
    # get ids of items in source and target domain
    i_ids = ratings["itemId"].unique()
    int_src_i_ids = list(range(len(i_ids)))
    i_string_to_id = dict(zip(i_ids, int_src_i_ids))
    ratings["itemId"] = ratings["itemId"].map(i_string_to_id)
    return i_string_to_id


def create_sim_matrix(
    paths_file_path: Path,
    src_i_string_to_id: dict[str, int],
    tgt_i_string_to_id: dict[str, int],
    src_n_items: int,
    tgt_n_items: int,
) -> csr_matrix:
    """
    Creates the similarity matrix between source items and target items based on the paths file.

    :param paths_file_path: path to the paths file
    :param src_i_string_to_id: mapping from item id to integer identifier for the source domain
    :param tgt_i_string_to_id: mapping from item id to integer identifier for the target domain
    :param src_n_items: number of source items
    :param tgt_n_items: number of target items
    :return: the sparse similarity matrix
    """
    # create source_items X target_items matrix (used for the Sim predicate in the model)
    logger.debug("Creating similarity matrix between source and target items using available paths")

    sparse_matrix = lil_matrix((src_n_items, tgt_n_items))
    for obj in orjsonl.stream(paths_file_path):
        n1_id = src_i_string_to_id[obj["n1_asin"]]
        n2_id = tgt_i_string_to_id[obj["n2_asin"]]
        sparse_matrix[n1_id, n2_id] = 1
    return sparse_matrix.tocsr()
