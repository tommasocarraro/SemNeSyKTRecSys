from src.data_preprocessing.Dataset import Domains_Type
from src.model_configs.mf.ModelConfigMf import HyperParamsMf

hyperparams_dict = {
    "movies": HyperParamsMf(
        n_factors=100, batch_size=512, learning_rate=0.0006856905672627633, weight_decay=0.09061932384191268
    ),
    "music": HyperParamsMf(
        n_factors=200, batch_size=512, learning_rate=0.0005219038500094297, weight_decay=0.09990886349746377
    ),
}


def get_mf_hyperparams(domain: Domains_Type) -> HyperParamsMf:
    """
    Retrieves the hyperparameters for the MF model

    :param domain: name of the training domain
    :return: hyperparams of the MF model
    """
    return hyperparams_dict[domain]
