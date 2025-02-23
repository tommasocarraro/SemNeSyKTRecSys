from src.model_configs.mf.ModelConfigMf import HyperParamsMf
from src.model_configs.utils import Domains_Type

hyperparams_dict = {
    "books": HyperParamsMf(n_factors=1, batch_size=64, learning_rate=0.01, weight_decay=0.01),
    "movies": HyperParamsMf(n_factors=1, batch_size=512, learning_rate=0.01, weight_decay=0.01),
    "music": HyperParamsMf(n_factors=1, batch_size=64, learning_rate=0.01, weight_decay=0.01),
}


def get_mf_hyperparams(domain: Domains_Type) -> HyperParamsMf:
    return hyperparams_dict[domain]
