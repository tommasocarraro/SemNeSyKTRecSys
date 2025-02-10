from loguru import logger

from src.model_configs.ModelConfig import LtnRegHyperParams
from src.model_configs.utils import Domains_Type

hyperparams_dict = {
    "books": {
        "movies": {
            0.05: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            0.2: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            0.5: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            1.0: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
        }
    },
    "movies": {
        "music": {
            0.05: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            0.2: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            0.5: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            1.0: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
        }
    },
    "music": {
        "movies": {
            0.05: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            0.2: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            0.5: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
            1.0: LtnRegHyperParams(
                n_factors=10,
                learning_rate=0.01,
                weight_decay=0.01,
                batch_size=256,
                p_forall_ax1=2,
                p_forall_ax2=2,
                p_sat_agg=2,
                neg_score=0.0,
                top_k_src=10,
            ),
        }
    },
}


def get_ltn_hyperparams(src_domain: Domains_Type, tgt_domain: Domains_Type, tgt_sparsity: float) -> LtnRegHyperParams:
    try:
        hyperparams = hyperparams_dict[src_domain][tgt_domain][tgt_sparsity]
    except KeyError:
        logger.error(f"Unsupported configuration: {src_domain}->{tgt_domain} @ {tgt_sparsity} sparsity")
        exit(1)
    return hyperparams
