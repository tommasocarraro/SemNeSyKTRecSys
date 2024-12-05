import numpy as np
import torch
from src import device


class DataLoader:
    """
    Data loader for the training of the MF model and the evaluation of all the models based on MF.

    It prepares the batches of user-item pairs to learn the MF model or evaluate all the models based on MF.
    """

    def __init__(self, data, ui_matrix, batch_size=1, shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training/evaluation triples (user, item, rating)
        :param ui_matrix: sparse user-item matrix of user interactions
        :param batch_size: batch size for the training/evaluation of the model
        :param shuffle: whether to shuffle data during training/evaluation or not
        """
        self.data = np.array(data)[data[:, -1] > 0]  # take only positive interaction for BPR loss
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ui_matrix = ui_matrix

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch_data = self.data[idxlist[start_idx:end_idx]]
            users = batch_data[:, 0]
            pos_items = batch_data[:, 1]
            # get specific portion of the user-item matrix
            batch_ui_matrix = self.ui_matrix[users]

            users = torch.tensor(users, dtype=torch.long)
            pos_items = torch.tensor(pos_items, dtype=torch.long)
            # sample negative items and avoid the sampling of positive items
            # neg_items = []
            # for u in batch_ui_matrix:
            #     neg_item = np.random.randint(0, self.ui_matrix.shape[1])
            #     while neg_item in u.nonzero()[1]:
            #         neg_item = np.random.randint(0, self.ui_matrix.shape[1])
            #     neg_items.append(neg_item)
            # neg_items = torch.tensor(neg_items, dtype=torch.long)
            neg_items = torch.randint(0, self.ui_matrix.shape[1], users.shape)

            yield users.to(device), pos_items.to(device), neg_items.to(device)
