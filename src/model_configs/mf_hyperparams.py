from loguru import logger

from src.model_configs.ModelConfig import MfHyperParams
from src.model_configs.utils import Domains_Type

hyperparams_dict = {
    "books": {
        0.05: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        0.2: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        0.5: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        1.0: MfHyperParams(
            n_factors=200, learning_rate=0.0001810210202954595, weight_decay=0.00026616161728996144, batch_size=256
        ),
    },
    "movies": {
        0.05: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        0.2: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        0.5: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        1.0: MfHyperParams(
            n_factors=200, learning_rate=0.00019603650397325152, weight_decay=0.07996262790412291, batch_size=512
        ),
    },
    "music": {
        0.05: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        0.2: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        0.5: MfHyperParams(n_factors=10, learning_rate=0.01, weight_decay=0.01, batch_size=256),
        1.0: MfHyperParams(
            n_factors=200, learning_rate=0.000404269153265299, weight_decay=0.07673355263885923, batch_size=512
        ),
    },
}


def get_mf_hyperparams(domain: Domains_Type, sparsity: float) -> MfHyperParams:
    try:
        hyperparams = hyperparams_dict[domain][sparsity]
    except KeyError:
        logger.error(f"Unsupported configuration: {domain} @ {sparsity} sparsity")
        exit(1)
    return hyperparams
