# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
