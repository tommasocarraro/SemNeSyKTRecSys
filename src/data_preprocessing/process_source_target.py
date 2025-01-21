import json
import os
from os import makedirs
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame
from scipy.sparse import csr_matrix

from src.utils import decompress_7z, set_seed
from .Dataset import Dataset
from .Split_Strategy import SplitStrategy
from src.model_configs.ModelConfig import DatasetConfig


def process_source_target(
    src_dataset_config: DatasetConfig,
    tgt_dataset_config: DatasetConfig,
    paths_file_path: Path,
    target_sparsity: float,
    seed: int,
    save_dir_path: Optional[Path] = None,
    clear_saved_dataset: bool = False,
) -> Dataset:
    """
    This function processes the cross-domain dataset and prepares it for the experiment. In particular, it executes
    the following pipeline:
    1. it creates train and validation sets for the source domain dataset
    2. it creates train, validation and test sets for the target domain dataset
    3. it creates the items_source X items_target matrix, containing 1 in position (i,j) if the source domain item i
    is semantically connected (there exists a KG paths) to the target domain item j, 0 otherwise

    :param src_dataset_config: source dataset configuration containing location and split strategy
    :param tgt_dataset_config: target dataset configuration containing location and split strategy
    :param paths_file_path: paths between entities in the two different domain
    :param target_sparsity: target domain sparsity factor
    :param save_dir_path: path where to save the dataset. None if the dataset should not be saved on disk.
    :param clear_saved_dataset: whether to clear the saved dataset if it exists
    """
    logger.info("Reading datasets from file system")
    # decompress source and target rating files, if needed
    src_dataset_path = decompress_7z(src_dataset_config.dataset_path)
    tgt_dataset_path = decompress_7z(tgt_dataset_config.dataset_path)

    if save_dir_path is not None:
        save_file_path = make_save_file_path(
            save_dir_path=save_dir_path,
            src_dataset_path=src_dataset_path,
            tgt_dataset_path=tgt_dataset_path,
            src_split_strategy=src_dataset_config.split_strategy,
            tgt_split_strategy=src_dataset_config.split_strategy,
            tgt_sparsity=target_sparsity,
        )

        maybe_dataset = load_or_clear_dataset(save_file_path=save_file_path, clear_saved_dataset=clear_saved_dataset)
        # if a precomputed dataset has been found, simply return it
        if maybe_dataset is not None:
            return maybe_dataset

    # otherwise obtain the dataframes containing the ratings for both domains
    src_ratings, tgt_ratings = load_dataframes(src_dataset_path=src_dataset_path, tgt_dataset_path=tgt_dataset_path)

    logger.debug("Applying transformations to the datasets")
    # reindex the users and items
    sh_users = set(src_ratings["userId"]) & set(tgt_ratings["userId"])
    sh_u_ids = list(range(len(sh_users)))
    sh_users_string_to_id = dict(zip(sorted(sh_users), sh_u_ids))
    reindex_users(
        ratings=src_ratings, sh_users=sh_users, sh_u_ids=sh_u_ids, sh_users_string_to_id=sh_users_string_to_id
    )
    src_i_string_to_id = reindex_items(ratings=src_ratings)
    reindex_users(
        ratings=tgt_ratings, sh_users=sh_users, sh_u_ids=sh_u_ids, sh_users_string_to_id=sh_users_string_to_id
    )
    tgt_i_string_to_id = reindex_items(ratings=tgt_ratings)

    # convert ratings to implicit feedback
    src_ratings["rating"] = (src_ratings["rating"] >= 4).astype(int)
    tgt_ratings["rating"] = (tgt_ratings["rating"] >= 4).astype(int)

    # get number of users and items in source and target domains
    src_n_users = src_ratings["userId"].nunique()
    src_n_items = src_ratings["itemId"].nunique()
    tgt_n_users = tgt_ratings["userId"].nunique()
    tgt_n_items = tgt_ratings["itemId"].nunique()

    # creating sparse user-item matrices for source and target domains
    sparse_src_matrix = create_ui_matrix(src_ratings, seed=seed)
    sparse_tgt_matrix = create_ui_matrix(tgt_ratings, seed=seed, sparsity_percent=target_sparsity)

    src_tr, src_val, src_te = src_dataset_config.split_strategy.split(src_ratings.to_numpy())
    tgt_tr, tgt_val, tgt_te = tgt_dataset_config.split_strategy.split(tgt_ratings.to_numpy())

    # create source_items X target_items matrix (used for the Sim predicate in the model)
    sim_matrix = create_sim_matrix(
        paths_file_path=paths_file_path,
        src_i_string_to_id=src_i_string_to_id,
        tgt_i_string_to_id=tgt_i_string_to_id,
        src_n_items=src_n_items,
        tgt_n_items=tgt_n_items,
    )

    dataset = Dataset(
        src_n_users=src_n_users,
        src_n_items=src_n_items,
        tgt_n_users=tgt_n_users,
        tgt_n_items=tgt_n_items,
        n_sh_users=len(sh_users),
        src_ui_matrix=sparse_src_matrix,
        tgt_ui_matrix=sparse_tgt_matrix,
        src_tr=src_tr,
        src_val=src_val,
        src_te=src_te,
        tgt_tr=tgt_tr,
        tgt_val=tgt_val,
        tgt_te=tgt_te,
        sim_matrix=sim_matrix,
    )

    if save_dir_path is not None:
        save_dataset(dataset, save_file_path)

    return dataset


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


def load_dataframes(src_dataset_path: Path, tgt_dataset_path: Path) -> tuple[DataFrame, DataFrame]:
    # get source and target ratings
    logger.debug("Reading the datasets' csv files with pandas")
    src_ratings = pd.read_csv(src_dataset_path, usecols=["userId", "itemId", "rating", "timestamp"])
    tgt_ratings = pd.read_csv(tgt_dataset_path, usecols=["userId", "itemId", "rating", "timestamp"])
    return src_ratings, tgt_ratings


def save_dataset(dataset: Dataset, save_file_path: Path) -> None:
    """
    Saves the dataset to the save_file_path

    :param dataset: dataset to save
    :param save_file_path: file path where the dataset should be saved
    """
    makedirs(save_file_path.parent, exist_ok=True)
    logger.info(f"Saving the dataset to file system")
    np.save(save_file_path, dataset)  # type: ignore


def create_ui_matrix(df: DataFrame, seed: int, sparsity_percent: float = 1.0) -> csr_matrix:
    """
    Creates the user-item interaction matrix from the given dataframe.

    :param df: dataframe containing the ratings
    :param seed: seed for the random number generator
    :param sparsity_percent: percent of ratings to keep in the dataset
    :return: a sparse matrix containing the user-item interactions
    """
    n_users = df["userId"].nunique()
    n_items = df["itemId"].nunique()

    if sparsity_percent is not None:
        if not (0 < sparsity_percent <= 1):
            raise ValueError("sparsity must be between 0 and 1, excluding zero.")

        if sparsity_percent < 1:
            logger.info(
                f"Artificially increasing the target domain's sparsity by retaining {sparsity_percent * 100}% of ratings for each user"
            )
            df = (
                df.groupby("userId")
                .apply(lambda x: x.sample(frac=sparsity_percent, random_state=seed))
                .reset_index(drop=True)
            )

    pos_ratings = df[df["rating"] == 1]
    return csr_matrix(
        (pos_ratings["rating"], (pos_ratings["userId"], pos_ratings["itemId"])), shape=(n_users, n_items)
    )


def make_save_file_path(
    save_dir_path: Path,
    src_dataset_path: Path,
    tgt_dataset_path: Path,
    tgt_sparsity: float,
    src_split_strategy: SplitStrategy,
    tgt_split_strategy: SplitStrategy,
) -> Path:
    """
    Constructs the path where the dataset should be stored on file system by numpy

    :param save_dir_path: path of the directory where the dataset should be stored
    :param src_dataset_path: path of the raw csv dataset
    :param tgt_dataset_path: path to the target dataset
    :param tgt_sparsity: the sparsity of the target dataset
    :param src_split_strategy: strategy used to split the source dataset
    :param tgt_split_strategy: strategy used to split the target dataset
    :return: Location of the processed dataset
    """
    makedirs(save_dir_path, exist_ok=True)

    source_file_name = f"{src_dataset_path.stem}_{tgt_dataset_path.stem}_sparsity={tgt_sparsity}_{hash(src_split_strategy)}_{hash(tgt_split_strategy)}.npy"
    return save_dir_path / source_file_name


def reindex_users(
    ratings: DataFrame, sh_users: set[int], sh_u_ids: list[int], sh_users_string_to_id: dict[str, int]
):
    """
    Re-indexes the users on the ratings dataframe in order to obtain incremental integer identifiers. Furthermore, the
    users shared between source and target domains are relocated to the first rows.

    :param ratings: the ratings dataframe
    :param sh_users: the set of shared users
    :param sh_u_ids: the list of shared user ids
    :param sh_users_string_to_id: the mapping from user_id string to integer identifier
    """
    # get ids of non-shared users in source and target domains
    users = set(ratings["userId"]) - sh_users
    u_ids = [(i + len(sh_u_ids)) for i in range(len(users))]
    u_string_to_id = dict(zip(sorted(users), u_ids))
    u_string_to_id.update(sh_users_string_to_id)
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

    # decompress the file containing the paths from source to target domain
    paths_file_path = decompress_7z(paths_file_path)

    with open(paths_file_path, "r") as json_paths:
        paths_file_path = json.load(json_paths)
    available_path_pairs = np.array(
        [
            (src_i_string_to_id[src_asin], tgt_i_string_to_id[tgt_asin])
            for src_asin, tgt_asins in paths_file_path.items()
            for tgt_asin, _ in tgt_asins.items()
        ]
    )

    # create sparse sim matrix
    return csr_matrix(
        (np.ones(available_path_pairs.shape[0]), (available_path_pairs[:, 0], available_path_pairs[:, 1])),
        shape=(src_n_items, tgt_n_items),
    )
