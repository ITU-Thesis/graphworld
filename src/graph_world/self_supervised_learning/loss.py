import torch
from torch import Tensor

# Small constant to avoid log(0)
err = 1e-15

def jensen_shannon_loss(positive_instance : Tensor, negative_instance : Tensor, reduction : str = 'mean') -> Tensor:
    '''
    Jensen-Shannon mutual information loss.
    '''
    assert (positive_instance >= 0).all() and (negative_instance >= 0).all()
    reducer = torch.mean
    if reduction == 'sum':
        reducer = torch.sum
    elif reduction == 'mean':
        reducer = torch.mean
    else:
        raise f'Could not find reduction {reduction} in jensen-shannon loss.'

    positive_MI = reducer((positive_instance + err).log())
    negative_MI = reducer(((1 - negative_instance) + err).log())
    return -(positive_MI + negative_MI)
