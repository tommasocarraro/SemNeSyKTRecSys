import argparse
from pathlib import Path
from typing import Optional

from ltn_usage import pretrain_mf_for_ltn, test_ltn, train_ltn, tune_ltn
from mf_usage import test_mf, train_mf, tune_mf
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

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument_group()
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
group.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, required=True, choices=["ltn", "mf"])
parser.add_argument("--src_dataset_name", type=str, choices=["movies", "music", "books"])
parser.add_argument("--tgt_dataset_name", type=str, choices=["movies", "music", "books"])
parser.add_argument("--clear_dataset", help="recompute dataset", action="store_true")
parser.add_argument("--sparsity_sh", help="sparsity factor for shared users", type=float, default=1.0)
parser.add_argument("--sweep_id", help="wandb sweep id", type=str, required=False)
parser.add_argument("--sweep_name", help="wandb sweep name", type=str, required=False)


save_dir_path = Path("data/saved_data/")

seed = 0


def main():
    args = parser.parse_args()

    src_dataset_name = args.src_dataset_name
    tgt_dataset_name = args.tgt_dataset_name
    clear_dataset: bool = args.clear_dataset
    sparsity_sh: float = args.sparsity_sh
    sweep_id: Optional[str] = args.sweep_id
    sweep_name: Optional[str] = args.sweep_name

    config = get_config_ltn(
        src_dataset_name=src_dataset_name, tgt_dataset_name=tgt_dataset_name, sparsity_sh=sparsity_sh, seed=seed
    )
    set_seed(seed)

    dataset = process_source_target(
        src_dataset_config=config.src_dataset_config,
        tgt_dataset_config=config.tgt_dataset_config,
        paths_file_path=config.paths_file_path,
        save_dir_path=save_dir_path,
        clear_saved_dataset=clear_dataset,
        sparsity_sh=sparsity_sh,
        seed=seed,
    )

    if args.model == "mf":
        dataset_comparison = get_dataset_comparison(dataset)
        config_comparison = get_config_mf(
            train_dataset_name=tgt_dataset_name, other_dataset_name=src_dataset_name, sparsity_sh=0.0, seed=seed
        )
        if args.train:
            train_mf(dataset=dataset_comparison, config=config_comparison)
        elif args.tune:
            tune_mf(dataset=dataset_comparison, config=config_comparison, sweep_id=sweep_id, sweep_name=sweep_name)
        else:
            test_mf(dataset=dataset_comparison, config=config_comparison)
    else:
        # pretrain the mf on the source domain with all ratings from shared users
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
        if args.train:
            # then train LTN on the target with no ratings from shared users
            train_ltn(dataset=target_dataset, config=config, processed_interactions=processed_interactions)
        elif args.tune:
            tune_ltn(
                dataset=target_dataset,
                config=config,
                processed_interactions=processed_interactions,
                sweep_name=sweep_name,
                sweep_id=sweep_id,
            )
        else:
            test_ltn(dataset=target_dataset, config=config, processed_interactions=processed_interactions)


if __name__ == "__main__":
    main()
