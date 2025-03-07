import ltn
import torch


class LTNLoss(torch.nn.Module):
    """
    Module for computing the Bayesian Personalized Ranking criterion.
    """

    def __init__(self):
        super(LTNLoss, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.Dist = ltn.Predicate(func=lambda pos, neg: self.sigmoid(pos - neg))

    # noinspection PyMethodMayBeStatic
    def forward(self, pos_scores: ltn.LTNObject, neg_scores: ltn.LTNObject) -> ltn.LTNObject:
        """
        It computes the Dist predicate of the LTN model. The Dist predicate takes a positive and negative scores as
        input and returns a measure of their distance. The objective of the model is to maximize this distance,
        similarly to the BPR.

        :param pos_scores: scores for the positive items computed by the Score function
        :param neg_scores: scores for the negative items computed by the Score function
        :return: averaged LTN loss
        """
        return self.Dist(pos_scores, neg_scores)
