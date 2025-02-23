import os
from pathlib import Path
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.data_preprocessing.Dataset import DatasetLtn, DatasetMf


def load_or_clear_dataset(save_file_path: Path, clear_saved_dataset: bool) -> Optional[Union[DatasetLtn, DatasetMf]]:
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


def save_dataset(dataset: Union[DatasetLtn, DatasetMf], save_file_path: Path) -> None:
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
        f"Selecting {sparsity_sh}% random ratings for each user"
    )
    tqdm.pandas(desc=desc, leave=False)

    def sample_ratings(x: DataFrameGroupBy):
        if x.name in sh_u_incr_ids:
            return x.sample(frac=sparsity_sh, random_state=seed)
        return x

    df = df.groupby("userId").progress_apply(sample_ratings).reset_index(drop=True)
    return df.to_numpy()


def remove_not_sh(ratings: NDArray, sh_u_ids: set[int]) -> NDArray:
    df = DataFrame(ratings, columns=["userId", "itemId", "rating"])
    df = df[df["userId"].isin(sh_u_ids)]
    return df.to_numpy()
