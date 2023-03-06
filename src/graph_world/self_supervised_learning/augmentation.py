import torch
from torch import Tensor
from typing import Union

def node_feature_shuffle(node_features : Tensor, edge_index : Tensor, seed : int = 987123, **kwargs) -> Union[Tensor, Tensor]:
    '''
    Shuffle the node features (rows) given a node feature matrix.
    '''
    assert edge_index.shape[0] == 2
    generator = torch.Generator()
    generator = generator.manual_seed(seed)
    augmented_node_features = node_features[torch.randperm(n=node_features.shape[0], generator=generator), :]
    return augmented_node_features, edge_index
