from os import makedirs
from pathlib import Path
from typing import Optional

import orjsonl
from loguru import logger
from pandas import DataFrame
from scipy.sparse import csr_matrix, lil_matrix

from src.utils import decompress_7z
from .Dataset import DatasetLtn
from .Split_Strategy import SplitStrategy
from .utils import (
    create_ui_matrix,
    increase_sparsity_sh,
    load_dataframes,
    load_or_clear_dataset,
    remove_not_sh,
    save_dataset,
)
from ..model_configs.CommonConfigs import DatasetConfig


def process_source_target_ltn(
    src_dataset_config: DatasetConfig,
    tgt_dataset_config: DatasetConfig,
    paths_file_path: Path,
    max_path_length: int,
    sparsity_sh: float = 1.0,
    save_dir_path: Optional[Path] = None,
    clear_saved_dataset: bool = False,
    seed: Optional[int] = None,
) -> DatasetLtn:
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
    :param sparsity_sh: target domain sparsity factor for shared users
    :param max_path_length: maximum path length to consider from the paths file
    :param save_dir_path: path where to save the dataset. None if the dataset should not be saved on disk.
    :param clear_saved_dataset: whether to clear the saved dataset if it exists
    :param seed: seed for sampling the dataset when increasing sparsity
    """
    logger.info("Reading datasets from file system")
    # decompress source and target rating files, if needed
    src_dataset_path = decompress_7z(src_dataset_config.dataset_path)
    tgt_dataset_path = decompress_7z(tgt_dataset_config.dataset_path)
    paths_file_path = decompress_7z(paths_file_path)

    save_file_path = None
    if save_dir_path is not None:
        save_file_path = make_save_file_path(
            save_dir_path=save_dir_path,
            src_dataset_path=src_dataset_path,
            tgt_dataset_path=tgt_dataset_path,
            src_split_strategy=src_dataset_config.split_strategy,
            tgt_split_strategy=src_dataset_config.split_strategy,
            tgt_n_ratings_sh=sparsity_sh,
            max_path_length=max_path_length,
            seed=seed,
        )

        maybe_dataset = load_or_clear_dataset(save_file_path=save_file_path, clear_saved_dataset=clear_saved_dataset)
        # if a precomputed dataset has been found, simply return it
        if maybe_dataset is not None:
            return maybe_dataset

    # otherwise obtain the dataframes containing the ratings for both domains
    src_ratings, tgt_ratings = load_dataframes(
        first_dataset_path=src_dataset_path, second_dataset_path=tgt_dataset_path
    )

    logger.debug("Applying transformations to the datasets")

    # reindex the users and items so they get incremental IDs, the shared users get the first IDs
    sh_users_amzn_ids: set[str] = set(src_ratings["userId"]) & set(tgt_ratings["userId"])
    sh_u_inc_ids = list(range(len(sh_users_amzn_ids)))
    sh_users_amzn_id_to_incr_id = dict(zip(sorted(sh_users_amzn_ids), sh_u_inc_ids))
    reindex_users(
        ratings=src_ratings,
        sh_users_amzn_ids=sh_users_amzn_ids,
        sh_u_inc_ids=sh_u_inc_ids,
        sh_users_amzn_id_to_incr_id=sh_users_amzn_id_to_incr_id,
    )
    src_i_string_to_id = reindex_items(ratings=src_ratings)
    reindex_users(
        ratings=tgt_ratings,
        sh_users_amzn_ids=sh_users_amzn_ids,
        sh_u_inc_ids=sh_u_inc_ids,
        sh_users_amzn_id_to_incr_id=sh_users_amzn_id_to_incr_id,
    )
    tgt_i_amzn_id_to_incr_id = reindex_items(ratings=tgt_ratings)

    # convert ratings to implicit feedback
    src_ratings["rating"] = (src_ratings["rating"] >= 4).astype(int)
    tgt_ratings["rating"] = (tgt_ratings["rating"] >= 4).astype(int)

    # get number of users and items in source and target domains
    src_n_users = src_ratings["userId"].nunique()
    src_n_items = src_ratings["itemId"].nunique()
    tgt_n_users = tgt_ratings["userId"].nunique()
    tgt_n_items = tgt_ratings["itemId"].nunique()

    # split the datasets into train, val, test and create the sparse interactions matrices
    logger.debug("Splitting the source dataset into train, val and test")
    src_tr, src_val, src_te = src_dataset_config.split_strategy.split(src_ratings.to_numpy())
    logger.debug("Creating the sparse interactions matrix for the source domain")
    sparse_src_matrix = create_ui_matrix(
        DataFrame(src_tr, columns=["userId", "itemId", "rating"]), n_users=src_n_users, n_items=src_n_items
    )

    logger.debug("Splitting the target dataset into train, val and test")
    tgt_tr, tgt_val, tgt_te = tgt_dataset_config.split_strategy.split(tgt_ratings.to_numpy())

    if sparsity_sh is not None:
        # remove ratings for shared users
        tgt_tr = increase_sparsity_sh(
            ratings=tgt_tr, sh_u_incr_ids=set(sh_u_inc_ids), seed=seed, sparsity_sh=sparsity_sh
        )
        tgt_val = increase_sparsity_sh(
            ratings=tgt_val, sh_u_incr_ids=set(sh_u_inc_ids), seed=seed, sparsity_sh=sparsity_sh
        )

    logger.debug("Creating the sparse interactions matrix for the target domain")
    sparse_tgt_matrix = create_ui_matrix(
        DataFrame(tgt_tr, columns=["userId", "itemId", "rating"]), n_users=tgt_n_users, n_items=tgt_n_items
    )

    tgt_te_sh = remove_not_sh(ratings=tgt_te, sh_u_ids=set(sh_u_inc_ids))

    # create source_items X target_items matrix (used for the Sim predicate in the model)
    sim_matrix = create_sim_matrix(
        paths_file_path=paths_file_path,
        src_i_string_to_id=src_i_string_to_id,
        tgt_i_string_to_id=tgt_i_amzn_id_to_incr_id,
        src_n_items=src_n_items,
        tgt_n_items=tgt_n_items,
        max_path_length=max_path_length,
    )

    dataset = DatasetLtn(
        src_n_users=src_n_users,
        src_n_items=src_n_items,
        tgt_n_users=tgt_n_users,
        tgt_n_items=tgt_n_items,
        n_sh_users=len(sh_users_amzn_ids),
        src_ui_matrix=sparse_src_matrix,
        tgt_ui_matrix=sparse_tgt_matrix,
        src_tr=src_tr,
        src_val=src_val,
        src_te=src_te,
        tgt_tr=tgt_tr,
        tgt_val=tgt_val,
        tgt_te=tgt_te,
        tgt_te_sh=tgt_te_sh,
        sim_matrix=sim_matrix,
        sh_users=set(sh_u_inc_ids),
    )

    if save_dir_path is not None and save_file_path is not None:
        save_dataset(dataset, save_file_path)

    return dataset


def make_save_file_path(
    save_dir_path: Path,
    src_dataset_path: Path,
    tgt_dataset_path: Path,
    tgt_n_ratings_sh: float,
    src_split_strategy: SplitStrategy,
    tgt_split_strategy: SplitStrategy,
    max_path_length: int,
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
    :param max_path_length: maximum path length to consider from the paths file
    :param seed: random seed
    :return: Location of the processed dataset
    """
    makedirs(save_dir_path, exist_ok=True)
    source_file_name = (
        f"src_dataset={src_dataset_path.stem}_tgt_dataset={tgt_dataset_path.stem}tgt_n_ratings_sh={tgt_n_ratings_sh}_"
        f"max_path_length={max_path_length}_{hash(src_split_strategy)}_{hash(tgt_split_strategy)}_seed={seed}.npy"
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
    logger.debug("Creating similarity matrix between source and target items using available paths")

    sparse_matrix = lil_matrix((src_n_items, tgt_n_items))
    for obj in orjsonl.stream(paths_file_path):
        n1_id = src_i_string_to_id[obj["n1_asin"]]
        n2_id = tgt_i_string_to_id[obj["n2_asin"]]
        if obj["path_length"] <= max_path_length:
            sparse_matrix[n1_id, n2_id] = 1
    return sparse_matrix.tocsr()
