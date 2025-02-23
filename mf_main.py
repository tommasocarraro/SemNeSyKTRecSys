import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from mf_usage import test_mf, train_mf, tune_mf
from src.data_preprocessing.process_source_target_mf import process_source_target_mf
from src.model_configs.mf.get_config_mf import get_config_mf
from src.utils import set_seed

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument_group()
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
group.add_argument("--test", action="store_true")
parser.add_argument("--train_dataset", type=str, choices=["movies", "music", "books"])
parser.add_argument("--other_dataset", type=str, choices=["movies", "music", "books"])
parser.add_argument("--clear_dataset", help="recompute dataset", action="store_true", required=False)
parser.add_argument("--sparsity_sh", help="TODO", type=float, required=False)  # TODO
parser.add_argument("--sweep_id", help="wandb sweep id", type=str, required=False)
parser.add_argument("--sweep_name", help="wandb sweep name", type=str, required=False)

save_dir_path = Path("data/saved_data/")

seed = 0


def main():
    args = parser.parse_args()

    train_dataset = args.train_dataset
    other_dataset = args.other_dataset
    clear_dataset: bool = args.clear_dataset
    sparsity_sh: Optional[float] = args.sparsity_sh
    sweep_id: Optional[str] = args.sweep_id
    sweep_name: Optional[str] = args.sweep_name

    if not args.tune and (sweep_id is not None or sweep_name is not None):
        logger.error("Sweep ID and sweep name should only be set when tuning")
        exit(1)

    train_config = get_config_mf(
        train_dataset_name=train_dataset, other_dataset_name=other_dataset, sparsity_sh=sparsity_sh, seed=seed
    )
    set_seed(seed)

    dataset = process_source_target_mf(
        train_dataset_config=train_config.train_dataset_config,
        other_dataset_config=train_config.other_dataset_config,
        save_dir_path=save_dir_path,
        clear_saved_dataset=clear_dataset,
        sparsity_sh=sparsity_sh,
        seed=seed,
    )

    if args.train:
        train_mf(dataset=dataset, config=train_config)
    elif args.tune:
        tune_mf(dataset=dataset, config=train_config, sweep_id=sweep_id, sweep_name=sweep_name)
    elif args.test:
        test_mf(dataset=dataset, config=train_config)


if __name__ == "__main__":
    main()
