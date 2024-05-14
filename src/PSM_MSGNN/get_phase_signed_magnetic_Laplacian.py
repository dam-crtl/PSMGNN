from typing import Optional
import numpy as np
import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_scipy_sparse_matrix,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse.linalg import eigsh
import warnings

from torch_geometric_signed_directed.utils import get_magnetic_signed_Laplacian

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

def get_phase_signed_magnetic_Laplacian(
    edge_index: torch.LongTensor,
    edge_weight: Optional[torch.Tensor] = None,
    normalization: Optional[str] = "sym",
    dtype: Optional[int] = None,
    num_nodes: Optional[int] = None,
    return_lambda_max: bool = False,
):
    r"""Computes the magnetic signed Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -
            1. :obj:`None`: No normalization :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`

            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`

        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)
        * **absolute_degree** (bool, optional) - Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)

    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic signed Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic signed Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic signed Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalization is not None:
        assert normalization in ["sym"], "Invalid normalization"

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=dtype, device=edge_index.device
        )

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    # calculate symmetrized matrix and skew symmetrized matrix
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    skew_sym_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)

    edge_attr = torch.stack([sym_attr, skew_sym_attr], dim=1)
    edge_index, syms_edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, "add")
    syms_edge_attr = syms_edge_attr / 2
    sym_edge_attr = syms_edge_attr[:, 0]
    skew_sym_edge_attr = syms_edge_attr[:, 1]

    #print(edge_index)
    #print(sym_edge_attr)
    #print(skew_sym_edge_attr)

    # calculate H
    H_weight = sym_edge_attr + skew_sym_edge_attr * 1j
    #print(H_weight)
    
    # calculate A
    squared_A_weight = sym_edge_attr.pow_(2) + skew_sym_edge_attr.pow_(2)
    A_weight = squared_A_weight.pow_(0.5)
    #print(A_weight)
    
    row, col = edge_index
    deg = scatter_add(A_weight, row, dim=0, dim_size=num_nodes)
    #print(deg)
    
    if normalization is None:
        # L = D_bar_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = torch.cat([-H_weight, deg], dim=0)
    elif normalization == "sym":
        # Compute H_norm = D_bar_sym^{-1/2} H D_bar_sym^{-1/2}
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = (deg_inv_sqrt[row] * H_weight * deg_inv_sqrt[col])
        # L = I - H_norm.
        edge_index, tmp = add_self_loops(
            edge_index, -edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
        assert tmp is not None
        edge_weight = tmp
    
    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        lambda_max = eigsh(L, k=1, which="LM", return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max


if __name__ == "__main__":
    values = torch.tensor([1.0, 6.0, 3.0, 4.0], dtype=torch.float32)
    row_indices = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    col_indices = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    coo_matrix = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]), values, size=(3, 3), device=device
    )

    edge_index = coo_matrix.coalesce().indices().to()
    edge_weight = coo_matrix.coalesce().values()

    edge_index1, edge_weight_real1, edge_weight_imag1, lambda_max1 = (
        get_phase_signed_magnetic_Laplacian(
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalization="sym",
            return_lambda_max=True,
        )
    )
    print(edge_index1)
    print(edge_weight_real1)
    print(edge_weight_imag1)
    print(lambda_max1)

    edge_index2, edge_weight_real2, edge_weight_imag2, lambda_max2 = (
        get_magnetic_signed_Laplacian(
            edge_index=edge_index, edge_weight=edge_weight, return_lambda_max=True
        )
    )
    print(edge_index2)
    print(edge_weight_real2)
    print(edge_weight_imag2)
    print(lambda_max2)

    """
    # 疎行列の要素を2乗する
    squared_values = coo_matrix.pow_(2)
    # 疎行列を転置
    transposed_values = coo_matrix.transpose()
    
    print(squared_values)
    print(transposed_values)
    """
