from src.data import process_source_target
from src.loader import DataLoader
from src.models.mf import MFTrainer, MatrixFactorization
import torch

process = process_source_target(0, "./data/ratings/reviews_Books_5.csv",
                                "./data/ratings/reviews_Movies_and_TV_5.csv",
                                "./data/kg_paths/books(pop:300)-movies(cs:5).json")

tr_loader = DataLoader(process["src_tr"], 128)
val_loader = DataLoader(process["src_val"], 128)

mf = MatrixFactorization(process["src_n_users"], process["src_n_items"], 10)

tr = MFTrainer(mf, torch.optim.Adam(mf.parameters(), lr=1e-3), torch.nn.MSELoss())

tr.train(tr_loader, val_loader, "mse")
