import torch
from torch import Tensor
from typing import Union

def node_feature_shuffle(node_features : Tensor, edge_index : Tensor, **kwargs) -> Union[Tensor, Tensor]:
    '''
    Shuffle the node features (rows) given a node feature matrix.
    '''
    assert edge_index.shape[0] == 2
    augmented_node_features = node_features[torch.randperm(n=node_features.shape[0]), :]
    return augmented_node_features, edge_index
