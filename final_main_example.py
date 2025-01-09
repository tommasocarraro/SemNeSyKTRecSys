from pathlib import Path

import torch
from loguru import logger

from src.cross_domain.ltn_trainer import LTNRegTrainer
from src.cross_domain.utils import get_reg_axiom_data
from src.data_loader import DataLoader, ValDataLoader
from src.data_preprocessing.process_source_target import process_source_target
from src.model import MatrixFactorization
from src.model_configs import get_config
from src.pretrain_source.inference import generate_pre_trained_src_matrix
from src.pretrain_source.loss import BPRLoss
from src.pretrain_source.mf_trainer import MfTrainer
from src.utils import set_seed

config = get_config(src_dataset_name="music", tgt_dataset_name="movies", kind="train")

dataset = process_source_target(
    src_dataset_config=config.src_dataset_config,
    tgt_dataset_config=config.tgt_dataset_config,
    paths_file_path=config.paths_file_path,
    save_dir_path=Path("./data/saved_data"),
)

set_seed(0)

mf_model_src = MatrixFactorization(
    n_users=dataset.src_n_users, n_items=dataset.src_n_items, n_factors=config.src_train_config.n_factors
)

mf_model_tgt = MatrixFactorization(
    n_users=dataset.tgt_n_users, n_items=dataset.tgt_n_items, n_factors=config.tgt_train_config.n_factors
)

trainer_src = MfTrainer(
    model=mf_model_src,
    optimizer=torch.optim.AdamW(
        params=mf_model_src.parameters(),
        lr=config.src_train_config.learning_rate,
        weight_decay=config.src_train_config.weight_decay,
    ),
    loss=BPRLoss(),
)

tr_loader_src = DataLoader(
    data=dataset.src_tr, ui_matrix=dataset.src_ui_matrix, batch_size=config.src_train_config.batch_size, n_negs=3
)
val_loader_src = ValDataLoader(
    data=dataset.src_val,
    ui_matrix=dataset.src_ui_matrix,
    batch_size=config.src_train_config.batch_size,
    sampled_n_negs=150,
    n_negs=100,
)
te_loader_src = ValDataLoader(
    data=dataset.src_te,
    ui_matrix=dataset.src_ui_matrix,
    batch_size=config.src_train_config.batch_size,
    sampled_n_negs=150,
    n_negs=100,
)

trainer_src.train(
    train_loader=tr_loader_src,
    val_loader=val_loader_src,
    val_metric=config.val_metric,
    early=config.early_stopping_patience,
    verbose=1,
    early_stopping_criterion=config.early_stopping_criterion,
    checkpoint_save_path=config.src_train_config.checkpoint_save_path,
    final_model_save_path=config.src_train_config.final_model_save_path,
)


test_metric, _ = trainer_src.validate(val_loader=te_loader_src, val_metric=config.val_metric)
logger.info(f"{config.val_metric.value} for source domain model computed on the test set: {test_metric}")

# trainer_tgt = LTNTrainer(
#     mf_model=mf_model_tgt,
#     optimizer=torch.optim.AdamW(
#         params=mf_model_tgt.parameters(),
#         lr=config.tgt_train_config.learning_rate,
#         weight_decay=config.tgt_train_config.weight_decay,
#     ),
# )

trainer_tgt = LTNRegTrainer(
    mf_model=mf_model_tgt,
    optimizer=torch.optim.AdamW(
        params=mf_model_tgt.parameters(),
        lr=config.tgt_train_config.learning_rate_range,
        weight_decay=config.tgt_train_config.weight_decay_range,
    ),
    processed_interactions=get_reg_axiom_data(
        src_ui_matrix=dataset.src_ui_matrix,
        tgt_ui_matrix=dataset.tgt_ui_matrix,
        n_sh_users=dataset.n_sh_users,
        sim_matrix=dataset.sim_matrix,
        top_k_items=generate_pre_trained_src_matrix(
            mf_model=mf_model_src,
            best_weights_path=config.src_train_config.final_model_save_path,
            n_shared_users=dataset.n_sh_users,
            top_k_src=20,
            batch_size=config.tgt_train_config.batch_size,
        ),
    ),
    p_forall=2,
    p_sat_agg=2,
)

tr_loader_tgt = DataLoader(
    data=dataset.tgt_tr, ui_matrix=dataset.tgt_ui_matrix, batch_size=config.tgt_train_config.batch_size, n_negs=3
)
val_loader_tgt = ValDataLoader(
    data=dataset.tgt_val,
    ui_matrix=dataset.tgt_ui_matrix,
    batch_size=config.tgt_train_config.batch_size,
    sampled_n_negs=150,
    n_negs=100,
)
te_loader_tgt = ValDataLoader(
    data=dataset.tgt_te,
    ui_matrix=dataset.tgt_ui_matrix,
    batch_size=config.tgt_train_config.batch_size,
    sampled_n_negs=150,
    n_negs=100,
)

trainer_tgt.train(
    train_loader=tr_loader_tgt,
    val_loader=val_loader_tgt,
    val_metric=config.val_metric,
    early=config.early_stopping_patience,
    verbose=1,
    early_stopping_criterion=config.early_stopping_criterion,
    checkpoint_save_path=config.tgt_train_config.checkpoint_save_path,
    final_model_save_path=config.tgt_train_config.final_model_save_path,
)

test_metric, _ = trainer_tgt.validate(val_loader=te_loader_tgt, val_metric=config.val_metric)
logger.info(f"{config.val_metric.value} for source domain model computed on the test set: {test_metric}")
