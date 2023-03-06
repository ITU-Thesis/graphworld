import torch
from torch import Tensor

# Small constant to avoid log(0)
err = 1e-15

def jensen_shannon_loss(positive_instance : Tensor, negative_instance : Tensor) -> Tensor:
    '''
    Jensen-Shannon mutual information loss.
    '''
    assert (positive_instance >= 0).all() and (negative_instance >= 0).all()
    positive_MI = (positive_instance + err).log().mean()
    negative_MI = (1 - negative_instance + err).log().mean()
    return -(positive_MI + negative_MI)
