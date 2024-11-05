from src.data import process_source_target
from src.loader import DataLoader
from src.models.mf import MFTrainerClassifier, MatrixFactorization
import torch

process = process_source_target(0, "./data/ratings/reviews_Books_5.csv",
                                "./data/ratings/reviews_Movies_and_TV_5.csv",
                                "./data/kg_paths/books(pop:300)-movies(cs:5).json",
                                save_path="./data/saved_data/dataset.npy")

tr_loader = DataLoader(process["src_tr"], 512)
val_loader = DataLoader(process["src_val"], 256)

mf = MatrixFactorization(process["src_n_users"], process["src_n_items"], 10)

tr = MFTrainerClassifier(mf, torch.optim.Adam(mf.parameters(), lr=1e-3), torch.nn.BCEWithLogitsLoss())

tr.train(tr_loader, val_loader, "fbeta-1", verbose=1)
