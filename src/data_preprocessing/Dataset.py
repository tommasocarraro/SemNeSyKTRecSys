from dataclasses import dataclass

from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from src.model_configs.utils import Domains_Type


@dataclass
class DatasetLtn:
    """
    Class which defines the structure of the object returned by process_source_target
    """

    src_dataset_name: Domains_Type
    tgt_dataset_name: Domains_Type
    src_n_users: int
    src_n_items: int
    tgt_n_users: int
    tgt_n_items: int
    n_sh_users: int
    src_ui_matrix: csr_matrix
    tgt_ui_matrix: csr_matrix
    src_tr: NDArray
    src_val: NDArray
    src_te: NDArray
    tgt_tr: NDArray
    tgt_val: NDArray
    sh_users: set[int]
    tgt_te: NDArray
    tgt_te_sh: NDArray
    sim_matrix: csr_matrix


@dataclass
class DatasetMf:
    """
    Class which defines the structure of the object returned by process_source_target
    """

    train_dataset_name: Domains_Type
    other_dataset_name: Domains_Type
    n_users: int
    n_items: int
    sh_users: set[int]
    ui_matrix: csr_matrix
    tr: NDArray
    val: NDArray
    te: NDArray
    te_sh: NDArray
