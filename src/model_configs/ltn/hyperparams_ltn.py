from loguru import logger

from src.model_configs.ltn.ModelConfigLtn import HyperParamsLtn
from src.model_configs.mf.ModelConfigMf import HyperParamsMf
from src.model_configs.utils import Domains_Type

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
                HyperParamsMf(n_factors=1, learning_rate=0.01, weight_decay=0.01, batch_size=512),
            )
        }
    },
    "movies": {
        "music": {
            0.0: (
                HyperParamsLtn(
                    n_factors=10,
                    learning_rate=0.01,
                    weight_decay=0.01,
                    batch_size=256,
                    p_forall_ax1=2,
                    p_forall_ax2=2,
                    p_sat_agg=2,
                ),
                HyperParamsMf(n_factors=1, learning_rate=0.01, weight_decay=0.01, batch_size=64),
            )
        }
    },
    "music": {
        "movies": {
            0.0: (
                HyperParamsLtn(
                    n_factors=10,
                    learning_rate=0.01,
                    weight_decay=0.01,
                    batch_size=256,
                    p_forall_ax1=2,
                    p_forall_ax2=2,
                    p_sat_agg=2,
                ),
                HyperParamsMf(n_factors=1, learning_rate=0.01, weight_decay=0.01, batch_size=64),
            )
        }
    },
}


def get_ltn_hyperparams(
    src_domain: Domains_Type, tgt_domain: Domains_Type, sparsity_sh: float
) -> tuple[HyperParamsLtn, HyperParamsMf]:
    try:
        hyperparams = hyperparams_dict[src_domain][tgt_domain][sparsity_sh]
    except KeyError:
        logger.error(f"Unsupported configuration: {src_domain}->{tgt_domain} @ {sparsity_sh} sparsity_sh")
        exit(1)
    return hyperparams
