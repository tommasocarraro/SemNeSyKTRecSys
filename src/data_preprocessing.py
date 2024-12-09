import json
from collections import defaultdict
from os import makedirs
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import py7zr
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split as train_test_split_sklearn


def train_test_split(
    seed: int,
    ratings: NDArray,
    frac: Optional[float] = None,
    user_level: bool = False,
):
    """
    It splits the dataset into training and test sets.

    :param seed: seed for random reproducibility
    :param ratings: numpy array containing the dataset ratings that have to be split
    :param frac: proportion of ratings to be sampled to create the test set. If None, it randomly samples one positive
    rating (LOO)
    :param user_level: whether the train test split has to be performed at the user level or ratings can
    be sampled randomly independently of the user. Defaults to False, meaning that the split is not done at the
    user level. In case the split is not done at the user level, a stratified split is performed to maintain the
    distribution of the classes
    :return: train and test set dataframes
    """
    if user_level:
        # Create a dictionary where each key is a user and the value is a list of indices for positive and negative
        # ratings
        user_indices_pos = defaultdict(list)
        user_indices_neg = defaultdict(list)

        for idx, user_id in enumerate(ratings[:, 0]):
            if ratings[idx, -1] == 1:
                user_indices_pos[user_id].append(idx)
            else:
                user_indices_neg[user_id].append(idx)

        # For each user, maintain the positive-negative rating distribution while sampling frac % for test set
        test_indices = []
        for user_id in user_indices_pos.keys():
            pos_indices = user_indices_pos[user_id]
            neg_indices = user_indices_neg[user_id]

            # Calculate sample size based on the number of positive ratings for this user
            if frac is not None:
                sample_size_pos = max(1, int(len(pos_indices) * frac))
                sample_size_neg = max(1, int(len(neg_indices) * frac))
            else:
                # when frac is None, it performs LOO sampling, so just one positive interaction for each user
                # is held out
                sample_size_pos = 1
                sample_size_neg = 1

            # sample is done if at least one positive interaction can remain in the training set
            if len(pos_indices) > 1:
                sampled_pos = np.random.choice(
                    pos_indices, size=sample_size_pos, replace=False
                )
                test_indices.extend(sampled_pos)
            # sample is done if at least one negative interaction can remain in the training set
            # if frac is None, it does LOO sampling, so the negatives are not sampled
            if len(neg_indices) > 1 and frac is not None:
                sampled_neg = np.random.choice(
                    neg_indices, size=sample_size_neg, replace=False
                )
                test_indices.extend(sampled_neg)

        # Create test set and training set based on the sampled indices
        test_set = ratings[test_indices]
        train_set = np.delete(ratings, test_indices, axis=0)
        return train_set, test_set
    else:
        assert (
            frac is not None
        ), "`frac` cannot be None if the split is not on the user level."
        # We use scikit-learn train-test split for the entire dataset
        return train_test_split_sklearn(
            ratings, random_state=seed, stratify=ratings[:, -1], test_size=frac
        )


def decompress_7z(compressed_file_path: Path):
    """
    It decompressed a given compressed file. If the file has no .7z extension, nothing is done by the function

    :param compressed_file_path: path to the compressed file
    :return: the path without the compression extension
    """
    # check if file exists
    if compressed_file_path.exists():
        # check if path is indeed pointing to a file
        if compressed_file_path.is_file():
            # splitting file the extension
            dirname = compressed_file_path.parent
            filename = compressed_file_path.stem
            extension = compressed_file_path.suffix
            output_path = dirname / filename
            if extension == ".7z":
                if not output_path.exists() or not output_path.is_file():
                    logger.debug(f"Decompressing {compressed_file_path}")
                    with py7zr.SevenZipFile(compressed_file_path, mode="r") as archive:
                        archive.extractall(path=compressed_file_path.parent)
            return output_path

        else:
            logger.error(
                f"Error. You can only decompress a file. Instead I got: {compressed_file_path}"
            )
            exit(1)
    else:
        logger.error(
            f"Trying to decompress a file which does not exist: {compressed_file_path}"
        )
        exit(1)


class SourceTargetDatasets(TypedDict):
    """
    Class which defines the structure of the object returned by process_source_target
    """

    src_n_users: int
    src_n_items: int
    tgt_n_users: int
    tgt_n_items: int
    src_ui_matrix: csr_array
    tgt_ui_matrix: csr_array
    src_tr: NDArray
    src_val: NDArray
    tgt_tr: NDArray
    tgt_tr_small: NDArray
    tgt_val: NDArray
    tgt_te: NDArray
    sim_matrix: csr_array


def process_source_target(
    seed: int,
    source_dataset_path: Path,
    target_dataset_path: Path,
    paths_file_path: Path,
    implicit: bool = True,
    source_val_size: float = 0.2,
    source_user_level_split: bool = True,
    target_val_size: float = 0.1,
    target_test_size: float = 0.2,
    target_user_level_split: bool = True,
    save_path: Optional[Path] = None,
) -> SourceTargetDatasets:
    """
    This function processes the cross-domain dataset and prepares it for the experiment. In particular, it executes
    the following pipeline:
    1. it creates train and validation sets for the source domain dataset
    2. it creates train, validation and test sets for the target domain dataset
    3. it creates the items_source X items_target matrix, containing 1 in position (i,j) if the source domain item i
    is semantically connected (there exists a KG paths) to the target domain item j, 0 otherwise

    :param seed: seed for reproducibility
    :param source_dataset_path: path to the source dataset
    :param target_dataset_path: path to the target dataset
    :param paths_file_path: paths between entities in the two different domain
    :param implicit: whether to convert explicit ratings to implicit feedback
    :param source_val_size: size of validation set for source domain dataset
    :param source_user_level_split: whether the split for the source dataset has to be done at the user level or not
    :param target_val_size: size of validation set for target domain dataset
    :param target_test_size: size of test set for target domain dataset
    :param target_user_level_split: whether the split for the target dataset has to be done at the user level or not
    :param save_path: path where to save the dataset. None if the dataset has no to be saved on disk.
    """
    logger.info("Reading datasets from file system")
    # decompress source and target rating files, if needed
    source_dataset_path = decompress_7z(source_dataset_path)
    target_dataset_path = decompress_7z(target_dataset_path)
    # computing the path where to store the processed datasets at
    actual_save_path = make_saved_dataset_path(
        save_path=save_path,
        source_dataset_path=source_dataset_path,
        target_dataset_path=target_dataset_path,
    )
    # if the dataset has already been processed before, simply load it with numpy
    if actual_save_path.is_file():
        logger.debug("Found precomputed dataset pair, reading it from file system")
        return np.load(actual_save_path, allow_pickle=True).item()
    # get source and target ratings
    logger.debug("Reading the datasets' csv files with pandas")
    src_ratings = pd.read_csv(
        source_dataset_path, usecols=["userId", "itemId", "rating"]
    )
    tgt_ratings = pd.read_csv(
        target_dataset_path, usecols=["userId", "itemId", "rating"]
    )

    logger.debug("Applying transformations to the datasets")
    # get shared users
    sh_users = set(src_ratings["userId"]) & set(tgt_ratings["userId"])

    # get ids of shared users
    sh_u_ids = list(range(len(sh_users)))
    sh_users_string_to_id = dict(zip(sh_users, sh_u_ids))

    # get ids of non-shared users in source and target domains
    src_users = set(src_ratings["userId"]) - sh_users
    src_u_ids = [(i + len(sh_u_ids)) for i in range(len(src_users))]
    src_u_string_to_id = dict(zip(src_users, src_u_ids))
    src_u_string_to_id.update(sh_users_string_to_id)
    tgt_users = set(tgt_ratings["userId"]) - sh_users
    tgt_u_ids = [(i + len(sh_u_ids)) for i in range(len(tgt_users))]
    tgt_u_string_to_id = dict(zip(tgt_users, tgt_u_ids))
    tgt_u_string_to_id.update(sh_users_string_to_id)

    # get ids of items in source and target domain
    src_i_ids = src_ratings["itemId"].unique()
    int_src_i_ids = list(range(len(src_i_ids)))
    src_i_string_to_id = dict(zip(src_i_ids, int_src_i_ids))
    tgt_i_ids = tgt_ratings["itemId"].unique()
    int_tgt_i_ids = list(range(len(tgt_i_ids)))
    tgt_i_string_to_id = dict(zip(tgt_i_ids, int_tgt_i_ids))

    # apply the new indexing
    src_ratings["userId"] = src_ratings["userId"].map(src_u_string_to_id)
    src_ratings["itemId"] = src_ratings["itemId"].map(src_i_string_to_id)
    tgt_ratings["userId"] = tgt_ratings["userId"].map(tgt_u_string_to_id)
    tgt_ratings["itemId"] = tgt_ratings["itemId"].map(tgt_i_string_to_id)

    if implicit:
        # convert ratings to implicit feedback
        src_ratings["rating"] = (src_ratings["rating"] >= 4).astype(int)
        tgt_ratings["rating"] = (tgt_ratings["rating"] >= 4).astype(int)

    # get number of users and items in source and target domains
    src_n_users = src_ratings["userId"].nunique()
    src_n_items = src_ratings["itemId"].nunique()
    tgt_n_users = tgt_ratings["userId"].nunique()
    tgt_n_items = tgt_ratings["itemId"].nunique()

    # creating sparse user-item matrices for source and target domains
    pos_src_ratings = src_ratings[src_ratings["rating"] == 1]
    pos_tgt_ratings = tgt_ratings[tgt_ratings["rating"] == 1]
    sparse_src_matrix = csr_array(
        (
            pos_src_ratings["rating"],
            (pos_src_ratings["userId"], pos_src_ratings["itemId"]),
        ),
        shape=(src_n_users, src_n_items),
    )
    sparse_tgt_matrix = csr_array(
        (
            pos_tgt_ratings["rating"],
            (pos_tgt_ratings["userId"], pos_tgt_ratings["itemId"]),
        ),
        shape=(tgt_n_users, tgt_n_items),
    )

    # create train and validation set for source domain dataset
    src_tr, src_val = train_test_split(
        seed,
        src_ratings.to_numpy(),
        frac=source_val_size,
        user_level=source_user_level_split,
    )

    # create train, validation and test set for target domain dataset
    tgt_tr, tgt_te = train_test_split(
        seed,
        tgt_ratings.to_numpy(),
        frac=target_test_size,
        user_level=target_user_level_split,
    )
    tgt_tr_small, tgt_val = train_test_split(
        seed, tgt_tr, frac=target_val_size, user_level=target_user_level_split
    )

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
    sim_matrix = csr_array(
        (
            np.ones(available_path_pairs.shape[0]),
            (available_path_pairs[:, 0], available_path_pairs[:, 1]),
        ),
        shape=(src_n_items, tgt_n_items),
    )

    dataset = {
        "src_n_users": src_n_users,
        "src_n_items": src_n_items,
        "tgt_n_users": tgt_n_users,
        "tgt_n_items": tgt_n_items,
        "src_ui_matrix": sparse_src_matrix,
        "tgt_ui_matrix": sparse_tgt_matrix,
        "src_tr": src_tr,
        "src_val": src_val,
        "tgt_tr": tgt_tr,
        "tgt_tr_small": tgt_tr_small,
        "tgt_val": tgt_val,
        "tgt_te": tgt_te,
        "sim_matrix": sim_matrix,
    }

    if save_path is not None:
        makedirs(actual_save_path.parent, exist_ok=True)
        logger.info(f"Saving the datasets pair to file system")
        np.save(actual_save_path, dataset)

    return dataset


def make_saved_dataset_path(
    save_path: Path, source_dataset_path: Path, target_dataset_path: Path
) -> Path:
    """
    Constructs the path where the dataset should be stored on file system by numpy

    :param save_path: path of the directory where the dataset should be stored
    :param source_dataset_path: path of the raw csv dataset
    :param target_dataset_path: path to the target dataset
    :return: Location of the processed dataset
    """

    if save_path.suffix != "":
        logger.error("save_path should be a directory")
        exit(1)
    makedirs(save_path, exist_ok=True)

    source_file_name = f"{source_dataset_path.stem}_{target_dataset_path.stem}.npy"
    return save_path / source_file_name
