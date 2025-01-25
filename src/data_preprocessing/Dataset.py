from dataclasses import dataclass

from numpy.typing import NDArray
from scipy.sparse import csr_matrix


@dataclass
class Dataset:
    """
    Class which defines the structure of the object returned by process_source_target
    """

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
    tgt_te: NDArray
    sim_matrix: csr_matrix
    tgt_true_negatives: NDArray
