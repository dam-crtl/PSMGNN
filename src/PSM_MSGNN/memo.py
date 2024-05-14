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

# def add_tensor(edge_index_a, edge_weight_a, edge_index_b, edge_weight_b):

"""

def calc_degree_matrix(adjacency_matrix):
    # adjacency_matrix: 疎行列
    # 各ノードの次数を計算
    degrees = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()
    # 次数行列を作成
    size = adjacency_matrix.size(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degree_matrix = torch.sparse_coo_tensor(
        torch.arange(size).unsqueeze(0).repeat(2, 1),
        degrees,
        size=(size, size),
        device=device,
        #dtype=torch.cfloat,
    )
    return degree_matrix
"""
"""
def powered_matrix(original_matrix, n):
    # original_matrix: 疎行列
    modified_matrix = original_matrix.coalesce()
    new_values = modified_matrix.values() ** n
    if n < 0:
        new_values[torch.isinf(new_values)] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    powered_matrix = torch.sparse_coo_tensor(
        modified_matrix.indices(),
        new_values,
        modified_matrix.size(),
        device=device,
        #dtype=torch.cfloat,
    )
    return powered_matrix
"""
"""
def sparse_identity_matrix(size):
    # size: 行列のサイズ
    # 単位行列を表現する疎行列を作成
    indices = torch.stack([torch.arange(size), torch.arange(size)])
    values = torch.ones(size, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_identity_matrix = torch.sparse_coo_tensor(
        indices, values, size=(size, size), device=device, #dtype=torch.cfloat
    )
    return sparse_identity_matrix
"""


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
    # sym_edge_index, sym_edge_attr = coalesce(edge_index, sym_edge_attr, num_nodes, num_nodes, "add")
    # skew_sym_edge_index, skew_sym_edge_attr = coalesce(edge_index, skew_sym_edge_attr, num_nodes, num_nodes, "add")
    # sym_edge_attr = sym_edge_attr / 2
    # skew_sym_edge_attr = skew_sym_edge_attr / 2
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
    """
    # caluculate H
    sym_row, sym_col = sym_edge_index[0], sym_edge_index[1]
    skew_sym_row, skew_sym_col = skew_sym_edge_index[0], skew_sym_edge_index[1]
    row = torch.cat([sym_row, skew_sym_row], dim=0)
    col = torch.cat([sym_col, skew_sym_col], dim=0)
    H_weight = torch.cat([sym_edge_attr, skew_sym_edge_attr * 1j], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    H_edge_index, H_edge_attr = coalesce(edge_index, H_weight, num_nodes, num_nodes, "add")

    squared_sym_edge_attr = sym_edge_attr.pow_(2)
    squared_skew_sym_edge_attr = skew_sym_edge_attr.pow_(2)
    squared_A_edge_attr = torch.cat([squared_sym_edge_attr, squared_skew_sym_edge_attr], dim=0)
    A_edge_index, squared_A_edge_attr = coalesce(edge_index, squared_A_edge_attr, num_nodes, num_nodes, "add")
    row, col = A_edge_index[0], A_edge_index[1]
    A_edge_attr = squared_A_edge_attr.pow_(0.5)
    
    deg = scatter_add(A_edge_attr, row, dim=0, dim_size=num_nodes)
    """
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
    
    """
    # ここまでで adjacency matrixを計算できている。
    A = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        dtype=torch.cfloat,
    )
    I = sparse_identity_matrix(num_nodes)

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency
    A_skew_sym = 0.5 * (A - A.T)

    # 以下の操作でvalueを使える
    # A_sym = A_sym.coalesce()
    # A_skew_sym = A_skew_sym.coalesce()

    # 各要素を2乗した行列を用意
    squared_A_sym = powered_matrix(A_sym, 2)
    squared_A_skew_sym = powered_matrix(A_skew_sym, 2)
    # 2つの行列をたす
    squared_true_A = squared_A_sym + squared_A_skew_sym

    # 真のAを求める
    true_A = powered_matrix(squared_true_A, 0.5)

    # 以下からグラフラプラシアンを計算する
    print(A_sym)
    H = A_sym + A_skew_sym * 1j
    H = H.to(device=device)
    # print(type(H))
    D = calc_degree_matrix(true_A)
    inv_sqrt_D = powered_matrix(D, -0.5)
    print(I)
    print(H)
    print(D)
    print(inv_sqrt_D)

    if normalization is None:
        L = D - H
    elif normalization == "sym":
        norm_H = inv_sqrt_D.mm(H).mm(inv_sqrt_D)
        L = I - norm_H

    L = L.to(device=device)
    # ここから最終回答へ
    edge_index = L.coalesce().indices()
    edge_weight = L.coalesce().values()

    # edge_index = edge_index
    # edge_weight.real = edge_weight.real
    # edge_weight.imag = float(edge_weight.imag)
    """

    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        lambda_max = eigsh(L, k=1, which="LM", return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max


if __name__ == "__main__":
    values = torch.tensor([1.0, 5.0, 3.0, 4.0], dtype=torch.float32)
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
