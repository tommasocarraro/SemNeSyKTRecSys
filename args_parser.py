import argparse
from typing import Literal

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--tune", action="store_true")
parser.add_argument(
    "datasets",
    type=str,
    help="Datasets to use (appears after --train or --tune)",
    nargs=2,
    choices=["movies", "music", "books"],
)
parser.add_argument("--src_model_path", type=str, help="Path to pretrained source model", required=False)
parser.add_argument("--clear", help="recompute dataset", action="store_true")
parser.add_argument("--sweep", help="wandb sweep id", type=str, required=False)
parser.add_argument(
    "--src_sparsity", help="sparsity factor of source dataset", type=float, required=False, default=1
)
parser.add_argument(
    "--tgt_sparsity", help="sparsity factor of target dataset", type=float, required=False, default=1
)


def get_args():
    args = parser.parse_args()

    if args.sweep and not args.tune:
        parser.error("--sweep can only be used with --tune")

    src_dataset_name, tgt_dataset_name = args.datasets
    src_model_path = args.src_model_path
    kind: Literal["train", "tune"] = "train" if args.train else "tune"
    sweep_id = args.sweep
    clear_dataset = args.clear
    src_sparsity = args.src_sparsity
    tgt_sparsity = args.tgt_sparsity

    return (
        src_dataset_name,
        tgt_dataset_name,
        src_model_path,
        kind,
        sweep_id,
        src_sparsity,
        tgt_sparsity,
        clear_dataset,
    )
