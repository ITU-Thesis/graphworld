from typing import Union
from torch import Tensor
import torch

def get_top_k_indices(input: Tensor, k : int, largest : bool = True) -> Union[Tensor, Tensor]:
    '''
    Get the indices of the top K elements.

    Params
    ------
    input:
        The tensor to find the top k elements
    k:
        The number of top elements to find
    largest:
        Whether to find the largest or smallest

    Returns
    -------
    (row_indices, col_indices) where input[row_indices, col_indices] returns the top k elements.

    '''
    assert input.dim() == 2
    N_cols = input.shape[1]
    
    top_k_indices = input.reshape(-1).topk(k=k, largest=largest).indices
    
    row = torch.div(top_k_indices, N_cols, rounding_mode='floor')
    col = top_k_indices % N_cols
    return row, col