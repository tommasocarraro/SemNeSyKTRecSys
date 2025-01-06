import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.device import device


class DataLoader:
    """
    Data loader for the training of the MF model and the evaluation of all the models based on MF.

    It prepares the batches of user-item pairs to learn the MF model or evaluate all the models based on MF.
    """

    def __init__(self, data, ui_matrix, batch_size=1, n_negs=3, shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training/evaluation triples (user, item, rating)
        :param ui_matrix: sparse user-item matrix of user interactions
        :param batch_size: batch size for the training/evaluation of the model
        :param n_negs: number of negative items that are sampled for each user. The more the number the more chances
        to do not sample positive items.
        :param shuffle: whether to shuffle data during training/evaluation or not
        """
        # take only positive interaction for BPR loss
        self.data = np.array(data)[data[:, 2] > 0]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ui_matrix = ui_matrix
        self.num_items = ui_matrix.shape[1]
        self.n_negs = n_negs

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
            # sample three negative items for each user in the batch
            neg_items = np.random.randint(0, self.num_items, (batch_data.shape[0] * self.n_negs,))
            # construct sparse matrix containing the negative items
            batch_ui_neg_matrix = csr_matrix(
                ([1] * neg_items.shape[0], (np.repeat(np.arange(0, batch_data.shape[0]), self.n_negs), neg_items)),
                batch_ui_matrix.shape,
            )
            # find the false negative items (items that are positive among the sampled negatives)
            false_neg_batch_ui_matrix = batch_ui_matrix.multiply(batch_ui_neg_matrix)

            # get mask for the sampled indices
            mask = 1 - false_neg_batch_ui_matrix[
                np.repeat(np.arange(0, batch_data.shape[0]), self.n_negs), neg_items
            ].reshape(-1, self.n_negs)
            # get final negative items
            neg_items = neg_items.reshape(-1, self.n_negs)
            neg_items = neg_items[np.arange(batch_data.shape[0]).reshape(-1, 1), np.argmax(mask, axis=1)]

            users = torch.tensor(users, dtype=torch.int32)
            pos_items = torch.tensor(pos_items, dtype=torch.int32)
            neg_items = torch.tensor(neg_items, dtype=torch.int32).squeeze(1)

            yield users.to(device), pos_items.to(device), neg_items.to(device)


class ValDataLoader:
    """
    Data loader for the training of the MF model and the evaluation of all the models based on MF.

    It prepares the batches of user-item pairs to learn the MF model or evaluate all the models based on MF.
    """

    def __init__(self, data, ui_matrix, batch_size=1, sampled_n_negs=3000, n_negs=100, shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training/evaluation triples (user, item, rating)
        :param ui_matrix: sparse user-item matrix of user interactions
        :param batch_size: batch size for the training/evaluation of the model
        :param sampled_n_negs: number of negative items that are sampled for each user. The more the number the more
        chances to do not sample positive items.
        :param n_negs: number of negative items that are keep from the sampled ones to construct the negative set to
        compute ranking metrics.
        :param shuffle: whether to shuffle data during training/evaluation or not
        """
        # take only positive interaction for BPR loss
        self.data = np.array(data)[data[:, 2] > 0]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ui_matrix = ui_matrix
        self.num_items = ui_matrix.shape[1]
        self.sampled_n_negs = sampled_n_negs
        self.n_negs = n_negs

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
            # sample sampled_n_negs negative items for each user in the batch
            neg_items = np.random.randint(0, self.num_items, (batch_data.shape[0] * self.sampled_n_negs,))
            # construct sparse matrix containing the negative items
            batch_ui_neg_matrix = csr_matrix(
                (
                    [1] * neg_items.shape[0],
                    (np.repeat(np.arange(0, batch_data.shape[0]), self.sampled_n_negs), neg_items),
                ),
                batch_ui_matrix.shape,
            )
            # find the false negative items (items that are positive among the sampled negatives)
            false_neg_batch_ui_matrix = batch_ui_matrix.multiply(batch_ui_neg_matrix)

            # get mask for the sampled indices
            mask = 1 - false_neg_batch_ui_matrix[
                np.repeat(np.arange(0, batch_data.shape[0]), self.sampled_n_negs), neg_items
            ].reshape(-1, self.sampled_n_negs)

            # get final negative items
            neg_items = neg_items.reshape(-1, self.sampled_n_negs)
            # use the argsort on the mask to get the indexes of the items that can be used as real negatives, namely
            # that are not false negatives
            # the argsort sorts in ascending order, so we need to be sure to take the last n_negs indices from each row
            neg_items = neg_items[
                np.arange(batch_data.shape[0]).repeat(self.n_negs).reshape(-1, self.n_negs),
                np.argsort(mask, axis=1)[:, -self.n_negs:],
            ]
            # create tensors
            users = torch.tensor(users, dtype=torch.int32)
            pos_items = torch.tensor(pos_items, dtype=torch.int32)
            neg_items = torch.tensor(neg_items, dtype=torch.int32)

            yield users.to(device), pos_items.to(device), neg_items.to(device)
