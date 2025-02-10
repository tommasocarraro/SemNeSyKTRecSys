from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


class SplitStrategy(ABC):
    @abstractmethod
    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        pass


@dataclass(frozen=True)
class LeaveLastOut(SplitStrategy):
    """
    This strategy splits the data such that both validation and test sets only contain the last positive interaction
    for each user, as long as there are any.
    """

    seed: Optional[int]

    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Computes the train, validation and test splits of the given ratings such that both validation and test splits
        contain only the last positive interaction for each user.

        :param ratings: the ratings numpy array
        :return: the train, val and test splits of the given ratings
        """
        train, test = _split_ratings(ratings, self.seed, lambda r: _compute_test_indices_one_out(r, "temporal"))
        train, val = _split_ratings(train, self.seed, lambda r: _compute_test_indices_one_out(r, "temporal"))
        return train, val, test


@dataclass(frozen=True)
class LeaveOneOut(SplitStrategy):
    """
    This strategy splits the data such that both validation and test sets only contain one random positive interaction
    for each user, as long as there are any.
    """

    seed: Optional[int]

    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Computes the train, validation and test splits of the given ratings such that both validation and test splits
        contain only one random positive interaction for each user.

        :param ratings: the ratings numpy array
        :return: the train, val and test splits of the given ratings
        """
        train, test = _split_ratings(ratings, self.seed, lambda r: _compute_test_indices_one_out(r, "sample"))
        train, val = _split_ratings(train, self.seed, lambda r: _compute_test_indices_one_out(r, "sample"))
        return train, val, test


@dataclass(frozen=True)
class PercentSplit(SplitStrategy):
    """
    Defines the percent split strategy, which means the dataset is partitioned into train, val and test using percentage
    values for validation and test.
    """

    seed: Optional[int]
    val_size: float
    test_size: float
    user_level: bool

    def split(self, ratings: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Computes the train, validation and test splits of the given ratings according to the percentage sizes of
        validation and test sets defined during initialization.

        :param ratings: the ratings numpy array
        :return: the train, val and test splits of the given ratings
        """
        assert self.val_size + self.test_size <= 1.0, "Validation + test percent size cannot be larger than 100"
        if self.user_level:
            train, test = _split_ratings(ratings, self.seed, lambda r: _compute_test_indices_frac(r, self.test_size))
            train, val = _split_ratings(train, self.seed, lambda r: _compute_test_indices_frac(r, self.val_size))
            return train, val, test
        else:
            train, test = train_test_split(
                ratings, random_state=self.seed, stratify=ratings[:, 2], test_size=self.test_size
            )
            train, val = train_test_split(
                train, random_state=self.seed, stratify=train[:, 2], test_size=self.val_size
            )
            return train, val, test


def _split_ratings(
    ratings: NDArray,
    seed: Optional[int],
    compute_test_indices: Callable[[NDArray], Union[list[int], tuple[list[int], NDArray]]],
) -> tuple[NDArray, NDArray]:
    """
    Higher order function which splits the given ratings into two sets by first computing the test indices through the
    callable parameter compute_test_indices, then removing the same indices from the ratings.

    :param ratings: the ratings numpy array
    :param seed: seed for reproducibility
    :param compute_test_indices: function to compute test indices based on ratings
    :return: the new train set and the fractionated set of ratings
    """
    # set numpy seed for reproducibility
    np.random.seed(seed)

    # compute the test indices
    test_indices = compute_test_indices(ratings)
    # if compute_test_indices returns a tuple, update the ratings array (temporal ordering)
    if isinstance(test_indices, tuple):
        test_indices, ratings = test_indices

    # create the new ratings sets through the test indices
    fractionated_set = ratings[test_indices]
    train_set = np.delete(ratings, test_indices, axis=0)
    return train_set, fractionated_set


def _get_user_ratings_indices(ratings: NDArray, kind: Literal["pos", "neg"]) -> dict[str, list[int]]:
    """
    Computes the ratings, either positive or negative, of all users contained in the ratings dataset.

    :param ratings: the ratings numpy array
    :param kind: the kind of ratings to compute, either "pos" or "neg"
    :return: a dictionary mapping users to the indices in the ratings dataset pointing to their positive interactions
    """
    user_indices = defaultdict(list)
    for idx, user_id in enumerate(ratings[:, 0]):
        if (kind == "pos" and ratings[idx, 2] == 1) or (kind == "neg" and ratings[idx, 2] == 0):
            user_indices[user_id].append(idx)
    return user_indices


def _compute_test_indices_one_out(
    ratings: NDArray, kind: Literal["temporal", "sample"]
) -> tuple[list[int], NDArray]:
    """
    Computes the test indices for all users contained in the ratings dataset such that each user only has one rating.
    If kind is "temporal", the last positive interaction is selected for each user, otherwise a random positive one
    is inserted in the test indices.

    :param ratings: the ratings numpy array
    :param kind: whether to select the last interaction or a random one for each user
    :return: a list containing the test indices and the ratings array, so that if the temporal ordering
    has been applied, the changes are propagated to the caller
    """
    # order ratings by timestamp, if necessary
    if kind == "temporal":
        ratings = ratings[ratings[:, -1].argsort()]

    # get each user's indices pointing to positive ratings
    user_indices_pos = _get_user_ratings_indices(ratings, "pos")

    # extract the test indices so that for each user we have either the last positive interaction or a sampled one
    test_indices = []
    for user_id in user_indices_pos.keys():
        pos_indices = user_indices_pos[user_id]

        # sample is done if at least one positive interaction can remain in the training set
        if len(pos_indices) > 1:
            test_index = pos_indices[-1] if kind == "temporal" else np.random.choice(pos_indices, replace=False)
            test_indices.append(test_index)
    return test_indices, ratings


def _compute_test_indices_frac(ratings: NDArray, frac: float) -> list[int]:
    """
    Computes the test indices for all users such that each user has frac interactions in the test indices, if any.

    :param ratings: the ratings numpy array
    :param frac: the fraction of interactions to select
    :return: a list containing the test indices
    """
    # obtain positive and negative interactions for each user
    user_indices_pos = _get_user_ratings_indices(ratings, "pos")
    user_indices_neg = _get_user_ratings_indices(ratings, "neg")

    # sample positive and negative interactions for each user according to frac
    test_indices = []
    for user_id in user_indices_pos.keys():
        pos_indices = user_indices_pos[user_id]
        sample_size_pos = max(1, int(len(pos_indices) * frac))
        if len(pos_indices) > 1:
            test_indices.extend(np.random.choice(pos_indices, size=sample_size_pos, replace=False))

        neg_indices = user_indices_neg[user_id]
        sample_size_neg = max(1, int(len(neg_indices) * frac))
        if len(neg_indices) > 1:
            test_indices.extend(np.random.choice(neg_indices, size=sample_size_neg, replace=False))
    return test_indices
