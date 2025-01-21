from typing import Callable, Literal

from .ModelConfig import ModelConfig
from .books_to_movies_configs import make_books_to_movies_config
from .movies_to_music_configs import make_movies_to_music_config
from .music_to_movies_configs import make_music_to_movies_config
from .utils import Domains_Type

configs_dict: dict[str, dict[str, Callable[[float, float, Literal["source", "target"]], ModelConfig]]] = {
    "books": {"movies": make_books_to_movies_config},
    "movies": {"music": make_movies_to_music_config},
    "music": {"movies": make_music_to_movies_config},
}


def get_config(
    src_dataset_name: Domains_Type,
    tgt_dataset_name: Domains_Type,
    src_sparsity: float,
    tgt_sparsity: float,
    which_dataset: Literal["source", "target"],
) -> ModelConfig:
    try:
        config_fn = configs_dict[src_dataset_name][tgt_dataset_name]
    except KeyError:
        raise ValueError("Unsupported dataset configuration")
    return config_fn(src_sparsity, tgt_sparsity, which_dataset)
