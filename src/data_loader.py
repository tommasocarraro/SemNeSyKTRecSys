from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch import Tensor
from loguru import logger
from src.device import device


class DataLoader(ABC):
    """
    Data loader for the training of the MF model and the evaluation of all the models based on MF.

    It prepares the batches of user-item pairs to learn the MF model or evaluate all the models based on MF.
    """

    data: NDArray
    shuffle: bool
    batch_size: int
    ui_matrix: csr_matrix
    num_items: int

    @abstractmethod
    def compute_neg_items(self, batch_data: NDArray, batch_ui_matrix: csr_matrix) -> Tensor:
        pass

    def _sample_neg_items(
        self, batch_data: NDArray, batch_ui_matrix: csr_matrix, sample_size: int
    ) -> tuple[NDArray, NDArray]:
        """
        Samples the required number of negative items from the given batch

        :param batch_data: batch of training data
        :param batch_ui_matrix: batch of the sparse user-item matrix of user interactions
        :param sample_size: number of negative items that are sampled for each user
        :return: the sampled negative items and the mask containing zeros where the sampled negatives ar false negatives
        and ones where the sampled negatives are true negatives
        """
        # sample negative items for each user in the batch
        neg_items = np.random.randint(0, self.num_items, (batch_data.shape[0] * sample_size,))
        # construct sparse matrix containing the negative items
        batch_ui_neg_matrix = csr_matrix(
            ([1] * neg_items.shape[0], (np.repeat(np.arange(0, batch_data.shape[0]), sample_size), neg_items)),
            batch_ui_matrix.shape,
        )
        # find the false negative items (items that are positive among the sampled negatives)
        false_neg_batch_ui_matrix = batch_ui_matrix.multiply(batch_ui_neg_matrix)

        # get mask for the sampled indices
        mask = 1 - false_neg_batch_ui_matrix[
            np.repeat(np.arange(0, batch_data.shape[0]), sample_size), neg_items
        ].reshape(-1, sample_size)

        neg_items = neg_items.reshape(-1, sample_size)

        return neg_items, mask

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            batch_data = self.data[idxlist[start_idx:end_idx]]
            users = batch_data[:, 0]
            pos_items = batch_data[:, 1]
            batch_ui_matrix = self.ui_matrix[users]

            neg_items = self.compute_neg_items(batch_data=batch_data, batch_ui_matrix=batch_ui_matrix)

            # create tensors
            users = torch.tensor(users, dtype=torch.int32)
            pos_items = torch.tensor(pos_items, dtype=torch.int32)

            yield users.to(device), pos_items.to(device), neg_items.to(device)


class TrDataLoader(DataLoader):
    def __init__(
        self,
        data: NDArray,
        ui_matrix: csr_matrix,
        batch_size=1,
        n_negs=3,
        shuffle=True,
        processed_interactions: Optional[dict[int, Tensor]] = None,
    ):
        """
        Constructor of the data loader.

        :param data: list of training/evaluation triples (user, item, rating)
        :param ui_matrix: sparse user-item matrix of user interactions
        :param batch_size: batch size for the training/evaluation of the model
        chances to do not sample positive items.
        :param n_negs: number of negative items that are keep from the sampled ones to construct the negative set to
        compute ranking metrics.
        :param shuffle: whether to shuffle data during training/evaluation or not
        """
        self.data = data[data[:, 2] > 0]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ui_matrix = ui_matrix
        self.num_items = ui_matrix.shape[1]
        self.n_negs = n_negs
        self.shuffle = shuffle
        self.processed_interactions = processed_interactions

        if processed_interactions is not None:
            self.sh_users = torch.tensor(list(processed_interactions.keys()), dtype=torch.int32, device=device)
            self.n_sh_users = self.sh_users.size()
            user_to_ratings: dict[int, set[int]] = defaultdict(set)
            for user, item in zip(*ui_matrix.nonzero()):
                if user in self.sh_users:
                    user_to_ratings[user].add(item)
            # compute the range of all item ids in the dataset
            all_items = np.arange(self.num_items)
            self.negative_candidates = defaultdict(Tensor)
            for user in processed_interactions.keys():
                c = np.setdiff1d(
                    all_items,
                    np.union1d(list(user_to_ratings[user]), processed_interactions[user].detach().cpu().numpy()),
                )
                self.negative_candidates[user] = torch.tensor(c, dtype=torch.int32, device=device)

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            batch_data = self.data[idxlist[start_idx:end_idx]]
            users = batch_data[:, 0]
            pos_items = batch_data[:, 1]
            batch_ui_matrix = self.ui_matrix[users]

            neg_items = self.compute_neg_items(batch_data=batch_data, batch_ui_matrix=batch_ui_matrix)

            # create tensors
            users = torch.tensor(users, dtype=torch.int32)
            pos_items = torch.tensor(pos_items, dtype=torch.int32)

            if self.processed_interactions is not None:
                # sample shared users
                indices = torch.randperm(len(self.sh_users))[: self.batch_size]
                sampled_sh_users = self.sh_users[indices]

                sampled_pos_items = torch.empty(size=(self.batch_size,), dtype=torch.int32)
                sampled_neg_items = torch.empty(size=(self.batch_size,), dtype=torch.int32)
                for user_idx, user in enumerate(sampled_sh_users):
                    user = user.item()  # type: ignore
                    pos = self.processed_interactions[user]
                    index_pos = torch.randint(0, len(pos), (1,))
                    sampled_pos_items[user_idx] = pos[index_pos]

                    neg = self.negative_candidates[user]
                    index_neg = torch.randint(0, len(neg), (1,))
                    sampled_neg_items[user_idx] = neg[index_neg]
                yield users.to(device), pos_items.to(device), neg_items.to(device), sampled_sh_users.to(
                    device
                ), sampled_pos_items.to(device), sampled_neg_items.to(device)
            else:
                yield users.to(device), pos_items.to(device), neg_items.to(device)

    def compute_neg_items(self, batch_data: NDArray, batch_ui_matrix: csr_matrix) -> Tensor:
        # sample n_negs negative items for each user in the batch
        neg_items, mask = self._sample_neg_items(
            batch_data=batch_data, sample_size=self.n_negs, batch_ui_matrix=batch_ui_matrix
        )
        neg_items = neg_items[np.arange(batch_data.shape[0]).reshape(-1, 1), np.argmax(mask, axis=1)]
        neg_items = torch.tensor(neg_items, dtype=torch.int32).squeeze()

        return neg_items


class ValDataLoader(DataLoader):
    def __init__(
        self,
        data: NDArray,
        ui_matrix: csr_matrix,
        n_sh_users: Optional[int] = None,
        batch_size=1,
        sampled_n_negs=150,
        n_negs=100,
        shuffle=True,
        test_only_sh_users=False,
    ):
        """
        Constructor of the data loader.

        :param sampled_n_negs: number of negative items that are sampled for each user. The more the number the more
        chances to do not sample positive items.
        :param n_negs: number of negative items that are keep from the sampled ones to construct the negative set to
        compute ranking metrics.
        :param test_only_sh_users: whether to only consider shared users during validation
        """
        self.data = data[data[:, 2] > 0]

        if test_only_sh_users:
            if n_sh_users is None:
                logger.error("Trying to validate using only shared users without passing the number of shared users")
                exit(1)
            sh_users = np.array(range(n_sh_users))
            mask = np.isin(data[:, 0], sh_users)
            self.data = data[mask]

        self.ui_matrix = ui_matrix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_negs = n_negs
        self.sampled_n_negs = sampled_n_negs
        self.num_items = ui_matrix.shape[1]

    def compute_neg_items(self, batch_data: NDArray, batch_ui_matrix: csr_matrix) -> Tensor:
        # sample sampled_n_negs negative items for each user in the batch
        neg_items, mask = self._sample_neg_items(
            batch_data=batch_data, sample_size=self.sampled_n_negs, batch_ui_matrix=batch_ui_matrix
        )
        # use the argsort on the mask to get the indexes of the items that can be used as real negatives, namely
        # that are not false negatives
        # the argsort sorts in ascending order, so we need to be sure to take the last n_negs indices from each row
        neg_items = neg_items[
            np.arange(batch_data.shape[0]).repeat(self.n_negs).reshape(-1, self.n_negs),
            np.argsort(mask, axis=1)[:, -self.n_negs :],
        ]
        neg_items = torch.tensor(neg_items, dtype=torch.int32)

        return neg_items
