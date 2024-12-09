import torch
import torch.nn as nn
from torch import Tensor


class BPRLoss(nn.Module):
    """
    Module for computing the Bayesian Personalized Ranking criterion.
    """

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_preds: Tensor, neg_preds: Tensor):
        """
        It computes the BPR criterion for the given positive and negative predictions.

        :param pos_preds: positive predictions
        :param neg_preds: negative predictions
        :return: averaged BPR loss
        """
        diff = pos_preds - neg_preds
        return -torch.mean(torch.log(diff.sigmoid()))
