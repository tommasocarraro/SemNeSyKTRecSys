from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray
from scipy.sparse import csr_matrix

Domains_Type = Literal["books", "movies", "music"]


@dataclass
class Dataset:
    """
    Data class which defines the structure of the object returned by process_source_target
    """

    src_dataset_name: Domains_Type
    tgt_dataset_name: Domains_Type
    src_n_users: int
    src_n_items: int
    tgt_n_users: int
    tgt_n_items: int
    sparsity_sh: float
    src_ui_matrix: csr_matrix
    tgt_ui_matrix: csr_matrix
    src_tr: NDArray
    src_val: NDArray
    src_te: NDArray
    src_te_sh: NDArray
    tgt_tr: NDArray
    tgt_val: NDArray
    tgt_te: NDArray
    tgt_te_sh: NDArray
    sh_users: set[int]
    n_sh_users: int
    sim_matrix: csr_matrix
    tgt_ui_matrix_no_sh: csr_matrix
    tgt_tr_no_sh: NDArray
    tgt_val_no_sh: NDArray


@dataclass
class DatasetPretrain:
    """
    Subset of Dataset used for training the source model required by LTN
    """

    n_users: int
    n_items: int
    ui_matrix: csr_matrix
    tr: NDArray
    val: NDArray
    te: NDArray


@dataclass
class DatasetTarget:
    """
    Subset of Dataset used for training LTN on the target domain
    """

    src_dataset_name: Domains_Type
    tgt_dataset_name: Domains_Type
    sparsity_sh: float
    n_users: int
    n_items: int
    n_sh_users: int
    tgt_tr_no_sh: NDArray
    tgt_val_no_sh: NDArray
    tgt_te: NDArray
    tgt_te_only_sh: NDArray
    src_ui_matrix: csr_matrix
    tgt_ui_matrix: csr_matrix
    tgt_ui_matrix_no_sh: csr_matrix
    sim_matrix: csr_matrix


@dataclass
class DatasetComparison:
    """
    Subset of Dataset used to train the BPR-MF model on the target domain
    """

    other_dataset_name: Domains_Type
    train_dataset_name: Domains_Type
    sparsity_sh: float
    n_users: int
    n_items: int
    tr_no_sh: NDArray
    val_no_sh: NDArray
    te: NDArray
    te_only_sh: NDArray
    ui_matrix: csr_matrix
    ui_matrix_no_sh: csr_matrix
