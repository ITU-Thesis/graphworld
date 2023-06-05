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

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple
import scipy as sp
import scipy.sparse as sprs
import scipy.sparse.linalg
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj, get_laplacian
from torch_geometric.transforms.gdc import GDC
from torch_geometric.utils import to_dense_adj

# Copied from https://github.com/Namkyeong/BGRL_Pytorch
class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new
    

def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

def pad_views(view1data, view2data):
        diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
        if diff > 0:
            smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
            smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
            view1data.x = F.normalize(view1data.x)
            view2data.x = F.normalize(view2data.x)


def k_closest_neighbors(data: Data, v : int, K : int) -> List[int]:
    '''
    Given a node v in a graph, find the neighbors that are closest to v in terms of connectivity.
    '''
    G = to_networkx(data)
    all_neighbors = nx.single_source_shortest_path_length(G=G, source=v, cutoff=K)
    sorted_neighbors = sorted(all_neighbors.items(), key=lambda x: x[1])
    
    return [x[0] for x in sorted_neighbors[0:K]]


def compute_InfoNCE_loss(z1: Tensor, z2: Tensor, tau: float = 1.0):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)

    refl_sim = torch.exp(torch.mm(z1, z1.t()) / tau) # inter-view
    between_sim = torch.exp(torch.mm(z1, z2.t()) / tau) # intra-view

    return -torch.log(
        between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )


def _check_input(
    x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tuple[Tensor, Tensor, bool]:
    """Check that input has the right dimensionality and sets the ``zero_diagonal`` argument if user has not
    provided import module.
    Args:
        x: tensor of shape ``[N,d]``
        y: if provided, a tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero
    """
    if x.ndim != 2:
        raise ValueError(f"Expected argument `x` to be a 2D tensor of shape `[N, d]` but got {x.shape}")

    if y is not None:
        if y.ndim != 2 or y.shape[1] != x.shape[1]:
            raise ValueError(
                "Expected argument `y` to be a 2D tensor of shape `[M, d]` where"
                " `d` should be same as the last dimension of `x`"
            )
        zero_diagonal = False if zero_diagonal is None else zero_diagonal
    else:
        y = x.clone()
        zero_diagonal = True if zero_diagonal is None else zero_diagonal
    return x, y, zero_diagonal


def _safe_matmul(x: Tensor, y: Tensor) -> Tensor:
    """Safe calculation of matrix multiplication.
    If input is float16, will cast to float32 for computation and back again.
    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return (x.float() @ y.T.float()).half()
    return x @ y.T


def pairwise_cosine_similarity(x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None):
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)

    norm = torch.norm(x, p=2, dim=1)
    x = x / norm.unsqueeze(1)
    norm = torch.norm(y, p=2, dim=1)
    y = y / norm.unsqueeze(1)

    distance = _safe_matmul(x, y)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance

def get_exact_ppr_matrix(data : Data, alpha: float) -> torch.Tensor:
    assert alpha >= 0. and alpha <= 1.
    data.edge_attr = None
    R = GDC(
        diffusion_kwargs={'alpha': 0.15, 'method': 'ppr'}, 
        sparsification_kwargs={'method':'threshold', 'avg_degree': data.num_edges // data.num_nodes}
    )(data.clone())
    return to_dense_adj(edge_index=R.edge_index, edge_attr=R.edge_attr, max_num_nodes=data.num_nodes).squeeze()
    
