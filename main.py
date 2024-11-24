import torch

from src.bpr_loss import BPRLoss
from src.data import process_source_target
from src.loader import DataLoader
from src.models.mf import MFTrainer, MatrixFactorization

process = process_source_target(
    0,
    "./data/ratings/reviews_Books_5.csv",
    "./data/ratings/reviews_Movies_and_TV_5.csv",
    "./data/kg_paths/books(pop:300)-movies(cs:5).json",
    save_path="./data/saved_data/dataset.npy",
)

tr_loader = DataLoader(process["src_tr"], process["src_n_items"], 512)
val_loader = DataLoader(process["src_val"], process["src_n_items"], 512)

mf = MatrixFactorization(process["src_n_users"], process["src_n_items"], 10)

tr = MFTrainer(
    mf, torch.optim.Adam(mf.parameters(), lr=1e-3, weight_decay=0.0001), BPRLoss()
)

tr.train(tr_loader, val_loader, "auc", verbose=1)
