from pathlib import Path
from typing import Optional

from loguru import logger
from pandas import DataFrame

from src.utils import decompress_7z
from .Dataset import Dataset, DatasetComparison, DatasetPretrain, DatasetTarget
from .utils import (
    create_sim_matrix,
    create_ui_matrix,
    increase_sparsity_sh,
    load_dataframes,
    load_or_clear_dataset,
    make_save_file_path,
    reindex_items,
    reindex_users,
    remove_not_sh,
    save_dataset,
)
from ..model_configs.CommonConfigs import DatasetConfig


def process_source_target(
    src_dataset_config: DatasetConfig,
    tgt_dataset_config: DatasetConfig,
    paths_file_path: Path,
    sparsity_sh: float = 1.0,
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
    :param sparsity_sh: target domain sparsity factor for shared users
    :param save_dir_path: path where to save the dataset. None if the dataset should not be saved on disk.
    :param clear_saved_dataset: whether to clear the saved dataset if it exists
    :param seed: seed for sampling the dataset when increasing sparsity
    """
    logger.info("Reading the LTN datasets from file system")
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
    src_tr_no_sh = increase_sparsity_sh(
        ratings=src_tr, sparsity_sh=sparsity_sh, sh_u_incr_ids=set(sh_u_inc_ids), seed=seed
    )
    src_te_sh = remove_not_sh(ratings=src_te, sh_u_ids=set(sh_u_inc_ids))
    logger.debug("Splitting the target dataset into train, val and test")
    tgt_tr, tgt_val, tgt_te = tgt_dataset_config.split_strategy.split(tgt_ratings.to_numpy())
    tgt_te_sh = remove_not_sh(ratings=tgt_te, sh_u_ids=set(sh_u_inc_ids))

    logger.debug("Creating the sparse interactions matrix for the source domain")
    sparse_src_matrix = create_ui_matrix(
        DataFrame(src_tr, columns=["userId", "itemId", "rating"]), n_users=src_n_users, n_items=src_n_items
    )
    sparse_src_matrix_no_sh = create_ui_matrix(
        DataFrame(src_tr_no_sh, columns=["userId", "itemId", "rating"]), n_users=src_n_users, n_items=src_n_items
    )

    # remove ratings for shared users
    tgt_tr_no_sh = increase_sparsity_sh(
        ratings=tgt_tr, sh_u_incr_ids=set(sh_u_inc_ids), seed=seed, sparsity_sh=sparsity_sh
    )
    tgt_val_no_sh = increase_sparsity_sh(
        ratings=tgt_val, sh_u_incr_ids=set(sh_u_inc_ids), seed=seed, sparsity_sh=sparsity_sh
    )

    logger.debug("Creating the sparse interactions matrix for the target domain")
    sparse_tgt_matrix = create_ui_matrix(
        DataFrame(tgt_tr, columns=["userId", "itemId", "rating"]), n_users=tgt_n_users, n_items=tgt_n_items
    )
    tgt_ui_matrix_no_sh = create_ui_matrix(
        DataFrame(tgt_tr_no_sh, columns=["userId", "itemId", "rating"]), n_users=tgt_n_users, n_items=tgt_n_items
    )

    # create source_items X target_items matrix (used for the Sim predicate in the model)
    sim_matrix = create_sim_matrix(
        paths_file_path=paths_file_path,
        src_i_string_to_id=src_i_string_to_id,
        tgt_i_string_to_id=tgt_i_amzn_id_to_incr_id,
        src_n_items=src_n_items,
        tgt_n_items=tgt_n_items,
    )

    dataset = Dataset(
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
        src_te_sh=src_te_sh,
        tgt_tr=tgt_tr,
        tgt_tr_no_sh=tgt_tr_no_sh,
        tgt_val=tgt_val,
        tgt_val_no_sh=tgt_val_no_sh,
        tgt_te=tgt_te,
        tgt_te_sh=tgt_te_sh,
        sim_matrix=sim_matrix,
        sh_users=set(sh_u_inc_ids),
        src_dataset_name=src_dataset_config.dataset_name,
        tgt_dataset_name=tgt_dataset_config.dataset_name,
        sparsity_sh=sparsity_sh,
        tgt_ui_matrix_no_sh=tgt_ui_matrix_no_sh,
    )

    if save_dir_path is not None and save_file_path is not None:
        save_dataset(dataset, save_file_path)

    return dataset


def get_pretrain_dataset(dataset: Dataset) -> DatasetPretrain:
    """
    Returns the dataset used for pretraining LTN's source model
    """
    return DatasetPretrain(
        n_users=dataset.src_n_users,
        n_items=dataset.src_n_items,
        ui_matrix=dataset.src_ui_matrix,
        tr=dataset.src_tr,
        val=dataset.src_val,
        te=dataset.src_te,
    )


def get_target_dataset(dataset: Dataset) -> DatasetTarget:
    """
    Returns the dataset used for training LTN
    """
    return DatasetTarget(
        n_users=dataset.tgt_n_users,
        n_items=dataset.tgt_n_items,
        n_sh_users=dataset.n_sh_users,
        tgt_tr_no_sh=dataset.tgt_tr_no_sh,
        tgt_val_no_sh=dataset.tgt_val_no_sh,
        tgt_te=dataset.tgt_te,
        tgt_te_only_sh=dataset.tgt_te_sh,
        src_ui_matrix=dataset.src_ui_matrix,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
        tgt_ui_matrix_no_sh=dataset.tgt_ui_matrix_no_sh,
        sim_matrix=dataset.sim_matrix,
        sparsity_sh=dataset.sparsity_sh,
        src_dataset_name=dataset.src_dataset_name,
        tgt_dataset_name=dataset.tgt_dataset_name,
    )


def get_dataset_comparison(dataset: Dataset) -> DatasetComparison:
    """
    Returns the dataset used for training the BPR-MF model on the target dataset
    """
    return DatasetComparison(
        n_users=dataset.tgt_n_users,
        n_items=dataset.tgt_n_items,
        ui_matrix=dataset.tgt_ui_matrix,
        ui_matrix_no_sh=dataset.tgt_ui_matrix_no_sh,
        tr_no_sh=dataset.tgt_tr_no_sh,
        val_no_sh=dataset.tgt_val_no_sh,
        te=dataset.tgt_te,
        te_only_sh=dataset.tgt_te_sh,
        train_dataset_name=dataset.tgt_dataset_name,
        other_dataset_name=dataset.src_dataset_name,
        sparsity_sh=dataset.sparsity_sh,
    )
