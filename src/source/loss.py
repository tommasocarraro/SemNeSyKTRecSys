import torch


class BPRLoss(torch.nn.Module):
    """
    Module for computing the Bayesian Personalized Ranking criterion.
    """

    def __init__(self):
        super(BPRLoss, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor):
        """
        It computes the BPR criterion for the given positive and negative predictions.

        :param pos_preds: positive predictions
        :param neg_preds: negative predictions
        :return: averaged BPR loss
        """
        diff = pos_preds - neg_preds
        return torch.nn.Softplus()(-diff).mean()
