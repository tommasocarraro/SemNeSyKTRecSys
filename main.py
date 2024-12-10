from pathlib import Path

import torch
from loguru import logger

from src.bpr_loss import BPRLoss
from src.configs import SWEEP_CONFIG_MF
from src.data_preprocessing import SourceTargetDatasets, process_source_target
from src.loader import DataLoader
from src.models.mf import MFTrainer, MatrixFactorization
from src.tuning import mf_tuning
from src.utils import set_seed


def main():
    dataset = process_source_target(
        seed=0,
        source_dataset_path=Path("./data/ratings/reviews_CDs_and_Vinyl_5.csv.7z"),
        target_dataset_path=Path("./data/ratings/reviews_Movies_and_TV_5.csv.7z"),
        paths_file_path=Path("./data/kg_paths/music(pop:200)->movies(cs:5).json.7z"),
        save_path=Path("./data/saved_data/"),
    )

    should_tune = False
    if should_tune:
        tune(dataset)
    else:
        train(dataset)


def train(dataset: SourceTargetDatasets):
    logger.info("Training the model...")
    set_seed(0)

    tr_loader = DataLoader(
        data=dataset["src_tr"], ui_matrix=dataset["src_ui_matrix"], batch_size=256
    )
    val_loader = DataLoader(
        data=dataset["src_val"], ui_matrix=dataset["src_ui_matrix"], batch_size=256
    )

    mf = MatrixFactorization(
        n_users=dataset["src_n_users"], n_items=dataset["src_n_items"], n_factors=25
    )

    tr = MFTrainer(
        mf_model=mf,
        optimizer=torch.optim.AdamW(mf.parameters(), lr=0.001, weight_decay=0.001),
        loss=BPRLoss(),
    )

    tr.train(
        train_loader=tr_loader,
        val_loader=val_loader,
        val_metric="auc",
        early=10,
        verbose=1,
    )

    # TODO exact sampling of negative without the risk of sampling positives


def tune(dataset: SourceTargetDatasets):
    mf_tuning(
        seed=0,
        tune_config=SWEEP_CONFIG_MF,
        train_set=dataset["src_tr"],
        val_set=dataset["src_val"],
        n_users=dataset["src_n_users"],
        n_items=dataset["src_n_items"],
        ui_matrix=dataset["src_ui_matrix"],
        metric="auc",
        entity_name="bmxitalia",
        exp_name="amazon",
    )


if __name__ == "__main__":
    main()
