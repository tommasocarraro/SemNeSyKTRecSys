import argparse
from pathlib import Path

from ltn_usage import pretrain_mf_for_ltn, test_ltn, train_ltn
from mf_usage import test_mf, train_mf
from src.cross_domain.utils import get_reg_axiom_data
from src.data_preprocessing.process_source_target import (
    get_dataset_comparison,
    get_pretrain_dataset,
    get_target_dataset,
    process_source_target,
)
from src.model_configs.ltn.get_config_ltn import get_config_ltn
from src.model_configs.mf.get_config_mf import get_config_mf
from src.pretrain_source.inference import generate_pre_trained_src_matrix
from src.utils import set_seed
import numpy as np
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int, help="test the model n times", required=False)
parser.add_argument("--model", type=str, required=True, choices=["ltn", "mf"])
parser.add_argument("--src_dataset_name", type=str, choices=["movies", "music", "books"])
parser.add_argument("--tgt_dataset_name", type=str, choices=["movies", "music", "books"])
parser.add_argument("--sparsity_sh", help="sparsity factor for shared users", type=float, default=1.0)

save_dir_path = Path("data/saved_data/")


def main():
    args = parser.parse_args()
    n = args.test
    src_dataset_name = args.src_dataset_name
    tgt_dataset_name = args.tgt_dataset_name
    model_name = args.model
    sparsity_sh = args.sparsity_sh

    set_seed(0)
    # generate list of seeds for testing
    rng = np.random.default_rng(0)
    seeds = rng.integers(low=1, high=10000000, size=n).tolist()

    metrics, metrics_sh = [], []
    for seed in seeds:
        config = get_config_ltn(
            src_dataset_name=src_dataset_name, tgt_dataset_name=tgt_dataset_name, sparsity_sh=sparsity_sh, seed=seed
        )
        set_seed(seed)
        dataset = process_source_target(
            src_dataset_config=config.src_dataset_config,
            tgt_dataset_config=config.tgt_dataset_config,
            paths_file_path=config.paths_file_path,
            save_dir_path=save_dir_path,
            clear_saved_dataset=False,
            sparsity_sh=sparsity_sh,
            seed=seed,
        )

        if model_name == "ltn":
            pretrain_dataset = get_pretrain_dataset(dataset)
            mf_pretrain = pretrain_mf_for_ltn(dataset=pretrain_dataset, config=config.src_model_config)

            top_k_items = generate_pre_trained_src_matrix(
                mf_model=mf_pretrain,
                best_weights_path=config.src_model_config.train_config.final_model_save_path,
                save_dir_path=save_dir_path,
                src_ui_matrix=pretrain_dataset.ui_matrix,
                n_shared_users=pretrain_dataset.n_users,
            )

            processed_interactions = get_reg_axiom_data(
                src_ui_matrix=dataset.src_ui_matrix,
                tgt_ui_matrix=dataset.tgt_ui_matrix_no_sh,
                n_sh_users=dataset.n_sh_users,
                sim_matrix=dataset.sim_matrix,
                top_k_items=top_k_items,
                save_dir_path=save_dir_path,
                src_dataset_name=src_dataset_name,
                tgt_dataset_name=tgt_dataset_name,
                sparsity_sh=sparsity_sh,
            )

            target_dataset = get_target_dataset(dataset)

            model_path = config.tgt_train_config.final_model_save_path
            if not model_path.is_file():
                logger.warning("Model not found. Training it now...")
                train_ltn(dataset=target_dataset, config=config, processed_interactions=processed_interactions)

            metric, metric_sh = test_ltn(
                dataset=target_dataset, config=config, processed_interactions=processed_interactions
            )
            metrics.append(metric)
            metrics_sh.append(metric_sh)
        else:
            dataset_comparison = get_dataset_comparison(dataset)
            config_comparison = get_config_mf(
                train_dataset_name=tgt_dataset_name, other_dataset_name=src_dataset_name, sparsity_sh=0.0, seed=seed
            )

            model_path = config_comparison.train_config.final_model_save_path
            if not model_path.is_file():
                logger.warning("Model not found. Training it now...")
                train_mf(dataset=dataset_comparison, config=config_comparison)

            metric, metric_sh = test_mf(dataset=dataset_comparison, config=config_comparison)
            metrics.append(metric)
            metrics_sh.append(metric_sh)
    logger.info(f"Average NDCG: {np.mean(metrics)}, average NDCG for shared users: {np.mean(metrics_sh)}")


if __name__ == "__main__":
    main()
