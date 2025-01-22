import argparse
from pathlib import Path
from typing import Literal, Optional

from src.data_preprocessing.process_source_target import process_source_target
from src.model_configs import get_config
from src.model_configs.utils import Domains_Type
from src.utils import set_seed
from loguru import logger
from train_mf import test_mf, train_mf, tune_mf
from train_ltn import test_ltn_reg, train_ltn_reg, tune_ltn_reg

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name", choices=["mf", "ltn_reg"])
parser.add_argument("--which_dataset", type=str, help="which dataset to use", choices=["source", "target"])
group2 = parser.add_mutually_exclusive_group(required=True)
parser.add_argument_group()
group2.add_argument("--train", action="store_true")
group2.add_argument("--tune", action="store_true")
group2.add_argument("--test", action="store_true")
parser.add_argument("datasets", type=str, help="Datasets to use", nargs=2, choices=["movies", "music", "books"])
parser.add_argument("--clear_dataset", help="recompute dataset", action="store_true")
parser.add_argument("--src_sparsity", help="sparsity factor of source dataset", type=float, required=False, default=1)
parser.add_argument("--tgt_sparsity", help="sparsity factor of target dataset", type=float, required=False, default=1)
parser.add_argument("--sweep_id", help="wandb sweep id", type=str, required=False)
parser.add_argument("--sweep_name", help="wandb sweep name", type=str, required=False)

save_dir_path = Path("data/saved_data/")


def main():
    args = parser.parse_args()
    model_name: Literal["mf", "ltn_reg"] = args.model
    which_dataset: Literal["source", "target"] = args.which_dataset

    if which_dataset == "source" and model_name == "ltn_reg":
        logger.error("Cannot train the ltn model with source dataset")
        exit(1)

    kind: Literal["train", "tune", "test"]
    if args.train:
        kind = "train"
    elif args.tune:
        kind = "tune"
    else:
        kind = "test"
    datasets: tuple[Domains_Type, Domains_Type] = args.datasets
    src_dataset_name, tgt_dataset_name = datasets
    clear_dataset: bool = args.clear_dataset
    src_sparsity: float = args.src_sparsity
    tgt_sparsity: float = args.tgt_sparsity
    sweep_id: Optional[str] = args.sweep_id
    sweep_name: Optional[str] = args.sweep_name

    config = get_config(
        src_dataset_name=src_dataset_name,
        tgt_dataset_name=tgt_dataset_name,
        src_sparsity=src_sparsity,
        tgt_sparsity=tgt_sparsity,
        which_dataset=which_dataset,
    )
    set_seed(config.seed)

    dataset = process_source_target(
        src_dataset_config=config.src_dataset_config,
        tgt_dataset_config=config.tgt_dataset_config,
        paths_file_path=config.paths_file_path,
        save_dir_path=save_dir_path,
        clear_saved_dataset=clear_dataset,
        src_sparsity=src_sparsity,
        tgt_sparsity=tgt_sparsity,
    )

    if kind == "train":
        if model_name == "mf":
            train_mf(dataset=dataset, config=config, which_dataset=which_dataset)
        else:
            train_ltn_reg(
                dataset=dataset,
                config=config,
                src_dataset_name=src_dataset_name,
                tgt_dataset_name=tgt_dataset_name,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                save_dir_path=save_dir_path,
            )
    elif kind == "tune":
        if model_name == "mf":
            tune_mf(
                dataset=dataset, config=config, which_dataset=which_dataset, sweep_id=sweep_id, sweep_name=sweep_name
            )
        else:
            tune_ltn_reg(
                dataset=dataset,
                config=config,
                src_dataset_name=src_dataset_name,
                tgt_dataset_name=tgt_dataset_name,
                tgt_sparsity=tgt_sparsity,
                sweep_id=sweep_id,
                sweep_name=sweep_name,
                save_dir_path=save_dir_path,
            )
    elif kind == "test":
        if model_name == "mf":
            test_mf(dataset=dataset, config=config, which_dataset=which_dataset)
        else:
            test_ltn_reg(
                dataset=dataset,
                config=config,
                src_dataset_name=src_dataset_name,
                tgt_dataset_name=tgt_dataset_name,
                src_sparsity=src_sparsity,
                tgt_sparsity=tgt_sparsity,
                save_dir_path=save_dir_path,
            )


if __name__ == "__main__":
    main()
