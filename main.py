import torch

from src.bpr_loss import BPRLoss
from src.data import process_source_target
from src.loader import DataLoader, DataLoaderSamuel
from src.models.mf import MFTrainer, MatrixFactorization
from src.utils import set_seed

process = process_source_target(
    0,
    "./data/ratings/reviews_CDs_and_Vinyl_5.csv",
    "./data/ratings/reviews_Movies_and_TV_5.csv",
    "./data/kg_paths/music(pop:200)->movies(cs:5).json.7z",
    save_path="./data/saved_data/",
)

set_seed(0)

tr_loader = DataLoaderSamuel(process["src_tr"], process["src_ui_matrix"], 512)
val_loader = DataLoader(process["src_val"], process["src_ui_matrix"], 512)

mf = MatrixFactorization(process["src_n_users"], process["src_n_items"], 5)

tr = MFTrainer(
    mf, torch.optim.AdamW(mf.parameters(), lr=0.01, weight_decay=0.0001), BPRLoss()
)

tr.train(tr_loader, val_loader, "auc", early=10, verbose=1)
