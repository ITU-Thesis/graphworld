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
