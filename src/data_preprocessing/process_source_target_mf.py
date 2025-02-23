from os import makedirs
from pathlib import Path
from typing import Optional

from loguru import logger
from pandas import DataFrame

from src.utils import decompress_7z
from .Dataset import DatasetMf
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


def process_source_target_mf(
    train_dataset_config: DatasetConfig,
    other_dataset_config: DatasetConfig,
    sparsity_sh: float = 1.0,
    save_dir_path: Optional[Path] = None,
    clear_saved_dataset: bool = False,
    seed: Optional[int] = None,
) -> DatasetMf:
    logger.info("Reading datasets from file system")
    # decompress source and target rating files, if needed
    train_dataset_path = decompress_7z(train_dataset_config.dataset_path)
    other_dataset_path = decompress_7z(other_dataset_config.dataset_path)

    save_file_path = None
    if save_dir_path is not None:
        save_file_path = make_save_file_path(
            save_dir_path=save_dir_path,
            sparsity_sh=sparsity_sh,
            seed=seed,
            train_dataset_path=train_dataset_path,
            other_dataset_path=other_dataset_path,
            split_strategy=train_dataset_config.split_strategy,
        )

        maybe_dataset = load_or_clear_dataset(save_file_path=save_file_path, clear_saved_dataset=clear_saved_dataset)
        # if a precomputed dataset has been found, simply return it
        if maybe_dataset is not None:
            return maybe_dataset

    # otherwise obtain the dataframes containing the ratings for both domains
    train_ratings, other_ratings = load_dataframes(
        first_dataset_path=train_dataset_path, second_dataset_path=other_dataset_path
    )

    logger.debug("Applying transformations to the datasets")

    # reindex the users and items so they get incremental IDs, the shared users get the first IDs
    sh_u_incr_ids = reindex_users(train_ratings=train_ratings, other_ratings=other_ratings)
    reindex_items(ratings=train_ratings)

    # convert ratings to implicit feedback
    train_ratings["rating"] = (train_ratings["rating"] >= 4).astype(int)

    # get number of users and items in source and target domains
    n_users = train_ratings["userId"].nunique()
    n_items = train_ratings["itemId"].nunique()

    # split the datasets into train, val, test and create the sparse interactions matrices
    logger.debug("Splitting the source dataset into train, val and test")
    tr, val, te = train_dataset_config.split_strategy.split(train_ratings.to_numpy())

    if sparsity_sh is not None:
        tr = increase_sparsity_sh(ratings=tr, sparsity_sh=sparsity_sh, sh_u_incr_ids=sh_u_incr_ids, seed=seed)
        val = increase_sparsity_sh(ratings=val, sparsity_sh=sparsity_sh, sh_u_incr_ids=sh_u_incr_ids, seed=seed)
    te_sh = remove_not_sh(ratings=te, sh_u_ids=sh_u_incr_ids)

    logger.debug("Creating the sparse interactions matrix for the source domain")
    uid_matrix = create_ui_matrix(
        DataFrame(tr, columns=["userId", "itemId", "rating"]), n_users=n_users, n_items=n_items
    )

    dataset = DatasetMf(
        n_users=n_users,
        n_items=n_items,
        sh_users=sh_u_incr_ids,
        ui_matrix=uid_matrix,
        tr=tr,
        val=val,
        te=te,
        te_sh=te_sh,
    )

    if save_dir_path is not None and save_file_path is not None:
        save_dataset(dataset, save_file_path)

    return dataset


def make_save_file_path(
    save_dir_path: Path,
    train_dataset_path: Path,
    other_dataset_path: Path,
    sparsity_sh: float,
    split_strategy: SplitStrategy,
    seed: int,
) -> Path:
    makedirs(save_dir_path, exist_ok=True)

    source_file_name = (
        f"train_dataset={train_dataset_path.stem}_other_dataset={other_dataset_path.stem}_"
        f"sparsity_sh={sparsity_sh}_{hash(split_strategy)}_seed={seed}.npy"
    )
    return save_dir_path / source_file_name


def reindex_users(train_ratings: DataFrame, other_ratings: DataFrame) -> set[int]:
    # get shared users
    train_u_amzn_ids = train_ratings["userId"].unique()
    other_u_amzn_ids = other_ratings["userId"].unique()
    sh_u_amzn_ids = set(train_u_amzn_ids).intersection(set(other_u_amzn_ids))

    # get ids of non-shared users in source and target domains
    u_amzn_id_to_incr_id = {amzn_id: incr_id for incr_id, amzn_id in enumerate(train_u_amzn_ids)}
    train_ratings["userId"] = train_ratings["userId"].map(u_amzn_id_to_incr_id)

    sh_u_incr_ids = set(u_amzn_id_to_incr_id[u] for u in sh_u_amzn_ids)

    return sh_u_incr_ids


def reindex_items(ratings: DataFrame):
    """
    Re-indexes the items on the ratings dataframe in order to obtain incremental integer identifiers.

    :param ratings: the ratings dataframe
    :return: the mapping from item id to integer identifier
    """
    # get ids of items in source and target domain
    i_ids = ratings["itemId"].unique()
    i_string_to_id = {amzn_id: incr_id for incr_id, amzn_id in enumerate(i_ids)}
    ratings["itemId"] = ratings["itemId"].map(i_string_to_id)
