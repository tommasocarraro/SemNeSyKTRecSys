from typing import Literal

from .ModelConfig import ModelConfig
from .books_to_movies_configs import train_books_to_movies_config, tune_books_to_movies_config
from .movies_to_books_configs import train_movies_to_books_config, tune_movies_to_books_config
from .movies_to_music_configs import train_movies_to_music_config, tune_movies_to_music_config
from .music_to_movies_configs import train_music_to_movies_config, tune_music_to_movies_config
from .utils import Domains_Type

configs_dict = {
    "books": {"movies": (train_books_to_movies_config, tune_books_to_movies_config)},
    "movies": {
        "books": (train_movies_to_books_config, tune_movies_to_books_config),
        "music": (train_movies_to_music_config, tune_movies_to_music_config),
    },
    "music": {"movies": (train_music_to_movies_config, tune_music_to_movies_config)},
}


def get_config(
    src_dataset_name: Domains_Type, tgt_dataset_name: Domains_Type, kind: Literal["train", "tune"]
) -> ModelConfig:
    try:
        config_pair = configs_dict[src_dataset_name][tgt_dataset_name]
    except KeyError:
        raise ValueError("Unsupported dataset configuration")

    if kind == "train":
        return config_pair[0]
    elif kind == "tune":
        return config_pair[1]
    else:
        raise ValueError("Unknown kind")
