import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from ltn_usage import test_ltn_reg, train_ltn_reg, tune_ltn_reg
from mf_usage import train_mf
from src.cross_domain.utils import get_reg_axiom_data
from src.data_preprocessing.process_source_target_ltn import process_source_target_ltn
from src.data_preprocessing.process_source_target_mf import process_source_target_mf
from src.model import MatrixFactorization
from src.model_configs.ltn.get_config_ltn import get_config_ltn
from src.model_configs.mf.get_config_mf import get_config_mf, make_mf_model_paths
from src.pretrain_source.inference import generate_pre_trained_src_matrix
from src.utils import set_seed

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument_group()
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
group.add_argument("--test", action="store_true")
parser.add_argument("--src_dataset_name", type=str, choices=["movies", "music", "books"])
parser.add_argument("--tgt_dataset_name", type=str, choices=["movies", "music", "books"])
parser.add_argument("--clear_dataset", help="recompute dataset", action="store_true", required=False)
parser.add_argument("--sparsity_sh", help="TODO", type=float, required=False)  # TODO
parser.add_argument("--max_path_length", type=int, help="maximum path length", required=True)
parser.add_argument("--sweep_id", help="wandb sweep id", type=str, required=False)
parser.add_argument("--sweep_name", help="wandb sweep name", type=str, required=False)

save_dir_path = Path("data/saved_data/")

seed = 0


def main():
    args = parser.parse_args()

    src_dataset_name = args.src_dataset_name
    tgt_dataset_name = args.tgt_dataset_name
    clear_dataset: bool = args.clear_dataset
    sparsity_sh: Optional[int] = args.sparsity_sh
    sweep_id: Optional[str] = args.sweep_id
    sweep_name: Optional[str] = args.sweep_name
    max_path_length: int = args.max_path_length

    if max_path_length and max_path_length < 0:
        logger.error("Max path length cannot be negative")
        exit(1)

    if not args.tune and (sweep_id is not None or sweep_name is not None):
        logger.error("Sweep ID and sweep name should only be set when tuning")
        exit(1)

    config = get_config_ltn(
        src_dataset_name=src_dataset_name, tgt_dataset_name=tgt_dataset_name, sparsity_sh=sparsity_sh, seed=seed
    )
    set_seed(seed)

    dataset = process_source_target_ltn(
        src_dataset_config=config.src_dataset_config,
        tgt_dataset_config=config.tgt_dataset_config,
        paths_file_path=config.paths_file_path,
        save_dir_path=save_dir_path,
        clear_saved_dataset=clear_dataset,
        sparsity_sh=sparsity_sh,
        seed=seed,
        max_path_length=max_path_length,
    )

    mf_model_src = MatrixFactorization(
        n_users=dataset.src_n_users,
        n_items=dataset.src_n_items,
        n_factors=config.train_config.hyper_params_source.n_factors,
    )

    src_model_weights_path = make_mf_model_paths(
        train_dataset_name=src_dataset_name, other_dataset_name=tgt_dataset_name, sparsity_sh=1.0, seed=seed
    )["final_model"]

    retrained_model = False
    if not src_model_weights_path or not src_model_weights_path.is_file():
        logger.warning(f"The source model '{src_model_weights_path}' does not exist. Training it now...")
        dataset_mf = process_source_target_mf(
            train_dataset_config=config.src_dataset_config,
            other_dataset_config=config.tgt_dataset_config,
            seed=seed,
            sparsity_sh=1.0,
            clear_saved_dataset=clear_dataset,
            save_dir_path=save_dir_path,
        )
        config_mf = get_config_mf(
            train_dataset_name=src_dataset_name, other_dataset_name=tgt_dataset_name, sparsity_sh=1.0, seed=seed
        )
        train_mf(dataset=dataset_mf, config=config_mf)
        retrained_model = True

    top_k_items = generate_pre_trained_src_matrix(
        mf_model=mf_model_src,
        best_weights_path=src_model_weights_path,
        n_shared_users=dataset.n_sh_users,
        save_dir_path=save_dir_path,
        src_ui_matrix=dataset.src_ui_matrix,
        retrained_model=retrained_model,
    )

    processed_interactions = get_reg_axiom_data(
        src_ui_matrix=dataset.src_ui_matrix,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
        n_sh_users=dataset.n_sh_users,
        sim_matrix=dataset.sim_matrix,
        top_k_items=top_k_items,
        save_dir_path=save_dir_path,
        src_dataset_name=src_dataset_name,
        tgt_dataset_name=tgt_dataset_name,
        sparsity_sh=sparsity_sh,
        retrained_model=retrained_model,
    )

    if args.train:
        train_ltn_reg(dataset=dataset, config=config, processed_interactions=processed_interactions)
    elif args.tune:
        tune_ltn_reg(
            dataset=dataset,
            config=config,
            processed_interactions=processed_interactions,
            sweep_id=sweep_id,
            sweep_name=sweep_name,
        )
    elif args.test:
        test_ltn_reg(dataset=dataset, config=config, processed_interactions=processed_interactions)


if __name__ == "__main__":
    main()
