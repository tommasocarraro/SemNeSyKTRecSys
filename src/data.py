import json
import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_sklearn
import numpy as np
from scipy.sparse import csr_array
import torch


def train_test_split(seed, ratings, frac=None, user_level=False):
    """
    It splits the dataset into training and test sets.

    :param seed: seed for random reproducibility
    :param ratings: dataframe containing the dataset ratings that have to be split
    :param frac: proportion of ratings to be sampled to create the test set. If None, it randomly sample one positive
    rating (LOO)
    :param user_level: whether the train test split has to be performed at the user level or ratings can
    be sampled randomly independently from the user. Defaults to False, meaning that the split is not done at the
    user level. In case the split is not done at the user level, a stratified split is performed to mantain the
    distribution of the classes
    :return: train and test set dataframes
    """
    if user_level:
        test_ids = ratings.groupby(by=["userId"]).apply(
            lambda x: x.sample(frac=frac, random_state=seed)
        )
        test_ids = test_ids.index.get_level_values(1)
        train_ids = np.setdiff1d(ratings.index.values, test_ids)
        train_set = ratings.iloc[train_ids].reset_index(drop=True)
        test_set = ratings.iloc[test_ids].reset_index(drop=True)
        return train_set, test_set
    else:
        # we use scikit-learn train-test split
        return train_test_split_sklearn(ratings, random_state=seed, stratify=ratings["sentiment"],
                                        test_size=frac)


def process_source_target(seed, source, target, paths, implicit=True, source_val_size=0.2,
                          source_user_level_split=True,
                          target_val_size=0.1, target_test_size=0.2, target_user_level_split=True):
    """
    This function processes the cross-domain dataset and prepares it for the experiment. In particular, it executes
    the following pipeline:
    1. it creates train and validation sets for the source domain dataset
    2. it creates train, validation and test sets for the target domain dataset
    3. it creates the items_source X items_target matrix, containing 1 in position (i,j) if the source domain item i
    is semantically connected (there exists a KG paths) to the target domain item j, 0 otherwise

    :param seed: seed for reproducibility
    :param source: path to the source dataset
    :param target: path to the target dataset
    :param paths: paths between entities in the two different domain
    :param implicit: whether to convert explicit ratings to implicit feedback
    :param source_val_size: size of validation set for source domain dataset
    :param source_user_level_split: whether the split for the source dataset has to be done at the user level or not
    :param target_val_size: size of validation set for target domain dataset
    :param target_test_size: size of test set for target domain dataset
    :param target_user_level_split: whether the split for the target dataset has to be done at the user level or not
    """
    # get source and target ratings
    src_ratings = pd.read_csv(source, usecols=["userId", "itemId", "rating"])
    tgt_ratings = pd.read_csv(target, usecols=["userId", "itemId", "rating"])
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
    # create train and validation set for source domain dataset
    src_tr, src_val = train_test_split(seed, src_ratings, frac=source_val_size, user_level=source_user_level_split)
    # create train, validation and test set for target domain dataset
    tgt_tr, tgt_te = train_test_split(seed, tgt_ratings, frac=target_test_size, user_level=target_user_level_split)
    tgt_tr_small, tgt_val = train_test_split(seed, tgt_tr, frac=target_val_size, user_level=target_user_level_split)

    # create source_items X target_items matrix (used for the Sim predicate in the model)
    with open(paths, "r") as json_paths:
        paths = json.load(json_paths)
    available_path_pairs = np.array([(src_asin, tgt_asin) for src_asin, tgt_asins in paths.items()
                                     for tgt_asin, _ in tgt_asins.items()])
    # create sparse sim matrix
    sim_matrix = csr_array((np.ones(available_path_pairs.shape[0]),
                            (available_path_pairs[:, 0], available_path_pairs[:, 1])),
                           shape=(src_n_items, tgt_n_items))

    return {"src_n_users": src_n_users, "src_n_items": src_n_items, "tgt_n_users": tgt_n_users,
            "tgt_n_items": tgt_n_items, "src_tr": src_tr.to_numpy(), "src_val": src_val.to_numpy(),
            "tgt_tr": tgt_tr.to_numpy(), "tgt_tr_small": tgt_tr_small.to_numpy(), "tgt_val": tgt_val.to_numpy(),
            "tgt_te": tgt_te.to_numpy(), "sim_matrix": torch.tensor(sim_matrix.toarray())}
