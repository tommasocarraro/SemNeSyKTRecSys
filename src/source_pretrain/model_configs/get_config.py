from typing import Literal
from .ModelConfig import ModelConfig
from .movies_configs import tune_movies_config, train_movies_config
from .music_configs import train_music_config, tune_music_config
from .books_configs import tune_books_config, train_books_config


def get_config(
    dataset_name: Literal["books", "movies", "music"], kind: Literal["train", "tune"]
) -> ModelConfig:
    if dataset_name == "books":
        config_pair = train_books_config, tune_books_config
    elif dataset_name == "movies":
        config_pair = train_movies_config, tune_movies_config
    elif dataset_name == "music":
        config_pair = train_music_config, tune_music_config
    else:
        raise ValueError("Unknown dataset")

    if kind == "train":
        return config_pair[0]
    elif kind == "tune":
        return config_pair[1]
    else:
        raise ValueError("Unknown kind")
