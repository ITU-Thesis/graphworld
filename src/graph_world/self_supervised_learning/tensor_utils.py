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

def repeat_rows(X : torch.tensor, n_repeats : int) -> torch.Tensor:
    '''
    Repeat the rows of a 2D tensor n_repeats times.

    Example
    -------
    [ 0 0 0 ]                        [ 0 0 0 ]
    [ 1 1 1 ] --> repeat 2 times --> [ 0 0 0 ]
    [ 2 2 2 ]                        [ 1 1 1 ]
                                     [ 1 1 1 ]
                                     [ 2 2 2 ]
                                     [ 2 2 2 ]
    '''
    assert X.dim() == 2
    return X.repeat(1, n_repeats).view(-1, X.shape[1])
