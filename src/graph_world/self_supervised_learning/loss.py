import torch
from torch import Tensor

def jensen_shannon_loss(positive_instance : Tensor, negative_instance : Tensor) -> Tensor:
    '''
    Jensen-Shannon mutual information loss.
    '''
    positive_loss = -(positive_instance).log().mean()
    negative_loss = -(1 - negative_instance).log().mean()
    return positive_loss + negative_loss
