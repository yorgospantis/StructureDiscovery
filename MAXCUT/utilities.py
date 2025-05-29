import torch
from itertools import product

def generate_adjacency_matrix(n: int, p: float = 0.5) -> torch.Tensor:
    adj = (torch.rand(n, n) < p).int()
    adj.fill_diagonal_(0)
    return adj.triu(1) + adj.triu(1).T

def max_cut(adjacency_matrix: torch.Tensor):
    n = adjacency_matrix.shape[0]
    partitions = torch.tensor(list(product([0, 1], repeat=n)))
    max_cut_value = float('-inf')
    best_partition = None

    for partition in partitions:
        set_1_mask = partition == 0
        set_2_mask = partition == 1
        cut_value = torch.sum(adjacency_matrix[set_1_mask][:, set_2_mask])
        if cut_value > max_cut_value:
            max_cut_value = cut_value.item()
            best_partition = partition.tolist()

    return max_cut_value, best_partition
