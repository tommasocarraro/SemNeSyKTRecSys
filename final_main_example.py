from pathlib import Path

import torch

import src.source_pretrain.loss
from src.data_preprocessing.process_source_target import process_source_target
from src.source_pretrain.data_loader import DataLoader, ValDataLoader
from src.source_pretrain.inference import generate_pre_trained_src_matrix
from src.source_pretrain.metrics import RankingMetricsType
from src.source_pretrain.model import MatrixFactorization
from src.model_configs import get_config
from src.source_pretrain.trainer import MfTrainer
from src.target.trainer import LTNRegTrainer
from src.target.utils import get_reg_axiom_data
from src.utils import set_seed

config = get_config(src_dataset_name="music", tgt_dataset_name="movies", kind="train")

dataset = process_source_target(
    src_dataset_config=config.src_dataset_config,
    tgt_dataset_config=config.tgt_dataset_config,
    paths_file_path=config.paths_file_path,
    save_dir_path=Path("./data/saved_data"),
)

set_seed(0)

mf_model_src = MatrixFactorization(dataset.src_n_users, dataset.src_n_items, 64)

mf_model_tgt = MatrixFactorization(dataset.tgt_n_users, dataset.tgt_n_items, 64)

trainer_src = MfTrainer(
    mf_model_src,
    torch.optim.AdamW(params=mf_model_src.parameters(), lr=0.001, weight_decay=0.00001),
    src.source_pretrain.loss.BPRLoss(),
)

tr_loader_src = DataLoader(dataset.src_tr[:1000], dataset.src_ui_matrix, 128, 3)
val_loader_src = ValDataLoader(dataset.src_val[:1000], dataset.src_ui_matrix, 512, 150, 100)
te_loader_src = ValDataLoader(dataset.src_te[:1000], dataset.src_ui_matrix, 512, 150, 100)

trainer_src.train(
    train_loader=tr_loader_src,
    val_loader=val_loader_src,
    val_metric=RankingMetricsType.NDCG10,
    early=5,
    verbose=1,
    early_stopping_criterion="val_metric",
    save_paths=(Path("./source_models/checkpoint_src_movies.pth"), Path("./source_models/best_src_movies.pth")),
)


te_ndcg, _ = trainer_src.validate(te_loader_src, RankingMetricsType.NDCG10)
print(te_ndcg)

# trainer = LTNTrainer(mf_model, torch.optim.AdamW(params=mf_model.parameters(), lr=0.001, weight_decay=0.00001))

trainer_tgt = LTNRegTrainer(
    mf_model_tgt,
    torch.optim.AdamW(params=mf_model_tgt.parameters(), lr=0.001, weight_decay=0.00001),
    get_reg_axiom_data(
        dataset.src_ui_matrix,
        dataset.tgt_ui_matrix,
        100,  # dataset.n_sh_users,
        dataset.sim_matrix,
        generate_pre_trained_src_matrix(
            mf_model_src,
            Path("./source_models/best_src_movies.pth"),
            100,  # dataset.n_sh_users,
            k=20,
            batch_size=512,
        ),
    ),
)

tr_loader_tgt = DataLoader(dataset.tgt_tr[:1000], dataset.tgt_ui_matrix, 128, 3)
val_loader_tgt = ValDataLoader(dataset.tgt_val[:1000], dataset.tgt_ui_matrix, 512, 150, 100)
te_loader_tgt = ValDataLoader(dataset.tgt_te[:1000], dataset.tgt_ui_matrix, 512, 150, 100)

trainer_tgt.train(
    train_loader=tr_loader_tgt,
    val_loader=val_loader_tgt,
    val_metric=RankingMetricsType.NDCG10,
    early=5,
    verbose=1,
    early_stopping_criterion="val_metric",
    save_paths=(Path("./source_models/checkpoint_src_books.pth"), Path("./source_models/best_src_books.pth")),
)

te_ndcg, _ = trainer_tgt.validate(te_loader_tgt, RankingMetricsType.NDCG10)
print(te_ndcg)
