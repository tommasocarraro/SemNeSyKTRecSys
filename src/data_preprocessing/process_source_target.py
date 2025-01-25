import orjson
import os
from os import makedirs
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.model_configs.ModelConfig import DatasetConfig
from src.utils import decompress_7z
from .Dataset import Dataset
from .Split_Strategy import SplitStrategy


def process_source_target(
    src_dataset_config: DatasetConfig,
    tgt_dataset_config: DatasetConfig,
    paths_file_path: Path,
    src_sparsity: float,
    tgt_sparsity: float,
    user_level_src: bool,
    user_level_tgt: bool,
    max_path_length: int,
    save_dir_path: Optional[Path] = None,
    clear_saved_dataset: bool = False,
    seed: Optional[int] = None,
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
    :param src_sparsity: source domain sparsity factor
    :param tgt_sparsity: target domain sparsity factor
    :param user_level_src: whether to sample sparsity% of each user's ratings or globally for the source domain
    :param user_level_tgt: whether to sample sparsity% of each user's ratings or globally for the target domain
    :param max_path_length: maximum path length to consider from the paths file
    :param save_dir_path: path where to save the dataset. None if the dataset should not be saved on disk.
    :param clear_saved_dataset: whether to clear the saved dataset if it exists
    :param seed: seed for sampling the dataset when increasing sparsity
    """
    logger.info("Reading datasets from file system")
    # decompress source and target rating files, if needed
    src_dataset_path = decompress_7z(src_dataset_config.dataset_path)
    tgt_dataset_path = decompress_7z(tgt_dataset_config.dataset_path)

    save_file_path = None
    if save_dir_path is not None:
        save_file_path = make_save_file_path(
            save_dir_path=save_dir_path,
            src_dataset_path=src_dataset_path,
            tgt_dataset_path=tgt_dataset_path,
            src_split_strategy=src_dataset_config.split_strategy,
            tgt_split_strategy=src_dataset_config.split_strategy,
            src_sparsity=src_sparsity,
            tgt_sparsity=tgt_sparsity,
            user_level_src=user_level_src,
            user_level_tgt=user_level_tgt,
            max_path_length=max_path_length,
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

    tgt_true_negatives = get_true_negatives(tgt_ratings)

    # convert ratings to implicit feedback
    src_ratings["rating"] = (src_ratings["rating"] >= 4).astype(int)
    tgt_ratings["rating"] = (tgt_ratings["rating"] >= 4).astype(int)

    # get number of users and items in source and target domains
    src_n_users = src_ratings["userId"].nunique()
    src_n_items = src_ratings["itemId"].nunique()
    tgt_n_users = tgt_ratings["userId"].nunique()
    tgt_n_items = tgt_ratings["itemId"].nunique()

    # creating sparse user-item matrices for source and target domains
    sparse_src_matrix = create_ui_matrix(src_ratings)
    sparse_tgt_matrix = create_ui_matrix(tgt_ratings)

    src_tr, src_val, src_te = src_dataset_config.split_strategy.split(src_ratings.to_numpy())
    src_tr = increase_sparsity(
        ratings=src_tr, sparsity=src_sparsity, label="source", seed=seed, user_level=user_level_src
    )
    tgt_tr, tgt_val, tgt_te = tgt_dataset_config.split_strategy.split(tgt_ratings.to_numpy())
    tgt_tr = increase_sparsity(
        ratings=tgt_tr, sparsity=tgt_sparsity, label="target", seed=seed, user_level=user_level_tgt
    )

    # create source_items X target_items matrix (used for the Sim predicate in the model)
    sim_matrix = create_sim_matrix(
        paths_file_path=paths_file_path,
        src_i_string_to_id=src_i_string_to_id,
        tgt_i_string_to_id=tgt_i_string_to_id,
        src_n_items=src_n_items,
        tgt_n_items=tgt_n_items,
        max_path_length=max_path_length,
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
        tgt_true_negatives=tgt_true_negatives,
    )

    if save_dir_path is not None and save_file_path is not None:
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


def create_ui_matrix(df: DataFrame) -> csr_matrix:
    """
    Creates the user-item interaction matrix from the given dataframe.

    :param df: dataframe containing the ratings
    :return: a sparse matrix containing the user-item interactions
    """
    n_users = df["userId"].nunique()
    n_items = df["itemId"].nunique()
    pos_ratings = df[df["rating"] == 1]
    return csr_matrix((pos_ratings["rating"], (pos_ratings["userId"], pos_ratings["itemId"])), shape=(n_users, n_items))


def make_save_file_path(
    save_dir_path: Path,
    src_dataset_path: Path,
    tgt_dataset_path: Path,
    src_sparsity: float,
    tgt_sparsity: float,
    src_split_strategy: SplitStrategy,
    user_level_src: bool,
    tgt_split_strategy: SplitStrategy,
    user_level_tgt: bool,
    max_path_length: int,
) -> Path:
    """
    Constructs the path where the dataset should be stored on file system by numpy

    :param save_dir_path: path of the directory where the dataset should be stored
    :param src_dataset_path: path of the raw csv dataset
    :param tgt_dataset_path: path to the target dataset
    :param src_sparsity: the sparsity of the source dataset
    :param tgt_sparsity: the sparsity of the target dataset
    :param src_split_strategy: strategy used to split the source dataset
    :param user_level_src: whether to sample sparsity% of each user's ratings or globally for the source domain
    :param tgt_split_strategy: strategy used to split the target dataset
    :param user_level_tgt: whether to sample sparsity% of each user's ratings or globally for the target domain
    :param max_path_length: maximum path length to consider from the paths file
    :return: Location of the processed dataset
    """
    makedirs(save_dir_path, exist_ok=True)

    source_file_name = (
        f"src_dataset={src_dataset_path.stem}_sparsity={src_sparsity}_ul={user_level_src}_"
        f"tgt_dataset={tgt_dataset_path.stem}_sparsity={tgt_sparsity}_ul={user_level_tgt}_"
        f"max_path_length={max_path_length}_{hash(src_split_strategy)}_{hash(tgt_split_strategy)}.npy"
    )
    return save_dir_path / source_file_name


def reindex_users(ratings: DataFrame, sh_users: set[int], sh_u_ids: list[int], sh_users_string_to_id: dict[str, int]):
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
    max_path_length: int,
) -> csr_matrix:
    """
    Creates the similarity matrix between source items and target items based on the paths file.

    :param paths_file_path: path to the paths file
    :param src_i_string_to_id: mapping from item id to integer identifier for the source domain
    :param tgt_i_string_to_id: mapping from item id to integer identifier for the target domain
    :param src_n_items: number of source items
    :param tgt_n_items: number of target items
    :param max_path_length: maximum path length to consider from the paths file
    :return: the sparse similarity matrix
    """
    # create source_items X target_items matrix (used for the Sim predicate in the model)
    logger.debug("Creating similarity matrix between source and target items, using available paths")

    # decompress the file containing the paths from source to target domain
    paths_file_path = decompress_7z(paths_file_path)

    with open(paths_file_path, "rb") as json_paths:
        paths_file = orjson.loads(json_paths.read())

    available_path_pairs = []
    for src_asin, tgt_asins in paths_file.items():
        for tgt_asin, paths in tgt_asins.items():
            if paths[0]["path_length"] <= max_path_length:
                available_path_pairs.append((src_i_string_to_id[src_asin], tgt_i_string_to_id[tgt_asin]))
    available_path_pairs = np.array(available_path_pairs)

    # create sparse sim matrix
    return csr_matrix(
        (np.ones(available_path_pairs.shape[0]), (available_path_pairs[:, 0], available_path_pairs[:, 1])),
        shape=(src_n_items, tgt_n_items),
    )


def increase_sparsity(
    ratings: NDArray, sparsity: float, label: Literal["source", "target"], seed: int, user_level: bool
) -> NDArray:
    """
    Artificially increases the sparsity of the ratings dataframe by the given sparsity factor. I.e., if sparsity is 0.4,
    then 40% of each user's ratings will be retained in the dataset. The ratings are sampled randomly.

    :param ratings: numpy array containing the quadruples of the ratings
    :param sparsity: the sparsity factor to use
    :param label: label representing either source or target domain, used for tqdm's description
    :param seed: seed to use for sampling
    :param user_level: whether to sample sparsity% of each user's ratings or globally

    :return: the new ratings array containing the sampled ratings

    """
    if not (0.0 < sparsity <= 1.0):
        logger.error("sparsity must be between 0 and 1, excluding 0.")
        exit(1)

    # if the sparsity is 1.0 then do nothing
    if sparsity == 1.0:
        return ratings

    # reconvert the numpy array to a dataframe
    df = DataFrame(ratings, columns=["userId", "itemId", "rating", "timestamp"])

    # sample the required percentage of ratings for each user
    # if after the sampling the user is left with no ratings, the original group of ratings is retained
    def sample_ratings(x):
        sampled = x.sample(frac=sparsity, random_state=seed)
        if len(sampled) > 0:
            return sampled
        return x

    if user_level:
        desc = (
            f"Artificially increasing data sparsity for the {label} domain. Selecting {int(sparsity * 100)}% random "
            f"ratings for each user"
        )
        # inject the tqdm methods to pandas
        tqdm.pandas(desc=desc)
        df = df.groupby("userId").progress_apply(sample_ratings).reset_index(drop=True)
    else:
        df = df.apply(sample_ratings).reset_index(drop=True)
    return df.to_numpy()


def get_true_negatives(ratings: DataFrame) -> NDArray:
    true_negatives = ratings[ratings["rating"] < 4.0]
    return true_negatives.to_numpy()
