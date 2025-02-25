from loguru import logger

from src.data_preprocessing.Dataset import Domains_Type
from src.model_configs.ltn.ModelConfigLtn import HyperParamsLtn
from src.model_configs.mf.ModelConfigMf import HyperParamsMf

books_pretrain_hyperparams = HyperParamsMf(
    n_factors=200, learning_rate=0.0001810210202954595, weight_decay=0.00026616161728996144, batch_size=256
)
movies_pretrain_hyperparams = HyperParamsMf(
    n_factors=200, learning_rate=0.00019603650397325152, weight_decay=0.07996262790412291, batch_size=512
)
music_pretrain_hyperparams = HyperParamsMf(
    n_factors=200, learning_rate=0.000404269153265299, weight_decay=0.07673355263885923, batch_size=512
)

hyperparams_dict = {
    "books": {
        "movies": {
            0.0: (
                HyperParamsLtn(
                    n_factors=1,
                    learning_rate=0.01,
                    weight_decay=0.01,
                    batch_size=512,
                    p_forall_ax1=2,
                    p_forall_ax2=2,
                    p_sat_agg=2,
                ),
                books_pretrain_hyperparams,
            )
        }
    },
    "movies": {
        "music": {
            0.0: (
                HyperParamsLtn(
                    n_factors=1,
                    learning_rate=0.01,
                    weight_decay=0.01,
                    batch_size=512,
                    p_forall_ax1=2,
                    p_forall_ax2=2,
                    p_sat_agg=2,
                ),
                movies_pretrain_hyperparams,
            )
        }
    },
    "music": {
        "movies": {
            0.0: (
                HyperParamsLtn(
                    n_factors=1,
                    learning_rate=0.01,
                    weight_decay=0.01,
                    batch_size=512,
                    p_forall_ax1=2,
                    p_forall_ax2=2,
                    p_sat_agg=2,
                ),
                music_pretrain_hyperparams,
            )
        }
    },
}


def get_ltn_hyperparams(
    src_domain: Domains_Type, tgt_domain: Domains_Type, sparsity_sh: float
) -> tuple[HyperParamsLtn, HyperParamsMf]:
    """
    Retrieves the hyperparameters for the LTN model with the given configuration

    :param src_domain: domain of the source domain
    :param tgt_domain: domain of the target domain
    :param sparsity_sh: target domain sparsity factor for shared users
    :return: Tuple of hyperparameters for the LTN model and its source MF model
    """
    try:
        hyperparams = hyperparams_dict[src_domain][tgt_domain][sparsity_sh]
    except KeyError:
        logger.error(f"Unsupported configuration: {src_domain}->{tgt_domain} @ {sparsity_sh} sparsity_sh")
        exit(1)
    return hyperparams
