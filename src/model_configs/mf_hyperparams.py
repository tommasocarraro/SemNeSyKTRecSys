from loguru import logger

from src.model_configs.ModelConfig import MfHyperParams
from src.model_configs.utils import Domains_Type

hyperparams_dict = {
    "books": {
        1.0: MfHyperParams(
            n_factors=200, learning_rate=0.0001810210202954595, weight_decay=0.00026616161728996144, batch_size=256
        )
    },
    "movies": {
        0.05: MfHyperParams(
            n_factors=1, learning_rate=0.00009256810357463333, weight_decay=0.0767257987641034, batch_size=128
        ),
        0.2: MfHyperParams(
            n_factors=5, learning_rate=0.0003068770535197875, weight_decay=0.06104282868347289, batch_size=128
        ),
        0.5: MfHyperParams(
            n_factors=200, learning_rate=0.0001415650617487978, weight_decay=0.08376300029107475, batch_size=256
        ),
        1.0: MfHyperParams(
            n_factors=200, learning_rate=0.00019603650397325152, weight_decay=0.07996262790412291, batch_size=512
        ),
    },
    "music": {
        0.05: MfHyperParams(
            n_factors=1, learning_rate=0.00004608565085495318, weight_decay=0.05480129294176802, batch_size=128
        ),
        0.2: MfHyperParams(
            n_factors=200, learning_rate=0.00011878251225203, weight_decay=0.07678144830103657, batch_size=182
        ),
        0.5: MfHyperParams(
            n_factors=200, learning_rate=0.00031360694935940074, weight_decay=0.06637606573142622, batch_size=128
        ),
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
