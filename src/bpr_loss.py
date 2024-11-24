import torch
import torch.nn as nn
from torch import Tensor


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_preds: Tensor, neg_preds: Tensor):
        diff = pos_preds - neg_preds
        return -torch.mean(torch.log(diff.sigmoid()))
