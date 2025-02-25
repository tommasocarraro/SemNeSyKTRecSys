from src.data_preprocessing.Dataset import Domains_Type
from src.model_configs.mf.ModelConfigMf import HyperParamsMf

hyperparams_dict = {
    "books": HyperParamsMf(n_factors=1, batch_size=64, learning_rate=0.01, weight_decay=0.01),
    "movies": HyperParamsMf(n_factors=1, batch_size=512, learning_rate=0.01, weight_decay=0.01),
    "music": HyperParamsMf(n_factors=1, batch_size=64, learning_rate=0.01, weight_decay=0.01),
}


def get_mf_hyperparams(domain: Domains_Type) -> HyperParamsMf:
    """
    Retrieves the hyperparameters for the MF model

    :param domain: name of the training domain
    :return: hyperparams of the MF model
    """
    return hyperparams_dict[domain]
