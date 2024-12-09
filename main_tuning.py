from src.data import process_source_target
from src.tuning import mf_tuning
from src.configs import SWEEP_CONFIG_MF

dataset = process_source_target(
    0,
    "./data/ratings/reviews_CDs_and_Vinyl_5.csv",
    "./data/ratings/reviews_Movies_and_TV_5.csv",
    "./data/kg_paths/music(pop:200)->movies(cs:5).json.7z",
    save_path="./data/saved_data/",
)


mf_tuning(
    0,
    SWEEP_CONFIG_MF,
    dataset["src_tr"],
    dataset["src_val"],
    dataset["src_n_users"],
    dataset["src_n_items"],
    "auc",
    exp_name="amazon",
    entity_name="bmxitalia",
)
