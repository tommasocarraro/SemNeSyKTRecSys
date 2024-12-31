from collections import defaultdict
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LeaveLastOut:
    # temporal split, last interaction in test set
    seed: int

    def _train_test_split(self, ratings: NDArray) -> tuple[NDArray, NDArray]:
        # set numpy seed for reproducibility
        np.random.seed(self.seed)
        # order ratings by timestamp, if necessary
        ratings = ratings[ratings[:, -1].argsort()]
        # Create a dictionary where each key is a user and the value is a list of indices for positive and negative
        # ratings
        user_indices_pos = defaultdict(list)

        for idx, user_id in enumerate(ratings[:, 0]):
            if ratings[idx, 2] == 1:
                user_indices_pos[user_id].append(idx)

        # For each user, maintain the positive-negative rating distribution while sampling frac % for test set
        test_indices = []
        for user_id in user_indices_pos.keys():
            pos_indices = user_indices_pos[user_id]

            # sample is done if at least one positive interaction can remain in the training set
            if len(pos_indices) > 1:
                sampled_pos = pos_indices[-1]
                test_indices.append(sampled_pos)

        # Create test set and training set based on the sampled indices
        test_set = ratings[test_indices]
        train_set = np.delete(ratings, test_indices, axis=0)
        return train_set, test_set

    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        train, test = self._train_test_split(ratings)
        train, val = self._train_test_split(train)
        return train, val, test


@dataclass(frozen=True)
class LeaveOneOut:
    # sampling of one positive interaction for test set
    seed: int

    def _train_test_split(self, ratings: NDArray) -> tuple[NDArray, NDArray]:
        # set numpy seed for reproducibility
        np.random.seed(self.seed)
        # Create a dictionary where each key is a user and the value is a list of indices for positive and negative
        # ratings
        user_indices_pos = defaultdict(list)

        for idx, user_id in enumerate(ratings[:, 0]):
            if ratings[idx, 2] == 1:
                user_indices_pos[user_id].append(idx)

        # For each user, maintain the positive-negative rating distribution while sampling frac % for test set
        test_indices = []
        for user_id in user_indices_pos.keys():
            pos_indices = user_indices_pos[user_id]

            # sample is done if at least one positive interaction can remain in the training set
            if len(pos_indices) > 1:
                sampled_pos = np.random.choice(pos_indices, replace=False)
                test_indices.append(sampled_pos)

        # Create test set and training set based on the sampled indices
        test_set = ratings[test_indices]
        train_set = np.delete(ratings, test_indices, axis=0)
        return train_set, test_set

    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        train, test = self._train_test_split(ratings)
        train, val = self._train_test_split(train)
        return train, val, test


@dataclass(frozen=True)
class PercentSplit:
    seed: int
    val_size: float = 0.2
    test_size: float = 0.2
    user_level: bool = True

    def _train_test_split(self, ratings: NDArray, frac: float) -> tuple[NDArray, NDArray]:
        # set numpy seed for reproducibility
        np.random.seed(self.seed)
        # Create a dictionary where each key is a user and the value is a list of indices for positive and negative
        # ratings
        user_indices_pos = defaultdict(list)
        user_indices_neg = defaultdict(list)

        for idx, user_id in enumerate(ratings[:, 0]):
            if ratings[idx, 2] == 1:
                user_indices_pos[user_id].append(idx)
            else:
                user_indices_neg[user_id].append(idx)

        # For each user, maintain the positive-negative rating distribution while sampling frac % for test set
        test_indices = []
        for user_id in user_indices_pos.keys():
            pos_indices = user_indices_pos[user_id]
            neg_indices = user_indices_neg[user_id]

            # Calculate sample size based on the number of positive ratings for this user
            sample_size_pos = max(1, int(len(pos_indices) * frac))
            sample_size_neg = max(1, int(len(neg_indices) * frac))

            # sample is done if at least one positive interaction can remain in the training set
            if len(pos_indices) > 1:
                sampled_pos = np.random.choice(pos_indices, size=sample_size_pos, replace=False)
                test_indices.extend(sampled_pos)
            # sample is done if at least one negative interaction can remain in the training set
            if len(neg_indices) > 1:
                sampled_neg = np.random.choice(neg_indices, size=sample_size_neg, replace=False)
                test_indices.extend(sampled_neg)

        # Create test set and training set based on the sampled indices
        test_set = ratings[test_indices]
        train_set = np.delete(ratings, test_indices, axis=0)
        return train_set, test_set

    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        if self.user_level:
            train, test = self._train_test_split(ratings, self.test_size)
            train, val = self._train_test_split(train, self.val_size)
            return train, val, test
        else:
            train, test = train_test_split(
                ratings, random_state=self.seed, stratify=ratings[:, 2], test_size=self.test_size
            )
            train, val = train_test_split(
                train, random_state=self.seed, stratify=train[:, 2], test_size=self.val_size
            )
            return train, val, test


@dataclass(frozen=True)
class SplitStrategy:
    src_split_strategy: Union[PercentSplit, LeaveOneOut, LeaveLastOut]
    tgt_split_strategy: Union[PercentSplit, LeaveOneOut, LeaveLastOut]
