from torch.nn import Linear, Bilinear
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn.functional as F
import gin
from .__types import *
from torch import Tensor
import torch_geometric.nn.models.autoencoder as pyg_autoencoder
from .basic_pretext_task import BasicPretextTask
from ...models.basic_gnn import SuperGAT
from torch_geometric.utils import negative_sampling, dropout_adj
from ..layers import NeuralTensorLayer
from typing import Tuple

# Based on https://github.com/CRIPAC-DIG/GRACE
@gin.configurable
class GRACE(BasicPretextTask):
    def __init__(self, 
                 tau : float = 0.5, 
                 edge_mask_ratio1 : float = 0.2,
                 edge_mask_ratio2 : float = 0.2,
                 feature_mask_ratio1 : float = 0.2, 
                 feature_mask_ratio2 : float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        # on the role of tau:
        # source: https://sslneurips21.github.io/files/CameraReady/tau_paper.pdf
        #----------------------------------------------------------------
        # A τ closer to 0 would accentuate when representations
        # are different, resulting in larger gradients. In the same vein,
        # a larger τ would be more forgiving of such differences.

        # GRACE specific hparams
        self.tau = tau
        self.edge_mask_ratio1 = edge_mask_ratio1
        self.edge_mask_ratio2 = edge_mask_ratio2
        self.feature_mask_ratio1 = feature_mask_ratio1
        self.feature_mask_ratio2 = feature_mask_ratio2

        # Decoder
        out = self.encoder.out_channels
        self.fc1 = Linear(out, out)
        self.fc2 = Linear(out, out)
        self.decoder = torch.nn.ModuleList([self.fc1, self.fc2])

    def generate_view(self, f_mask_ratio : float, e_mask_ratio : float) -> Tuple[Tensor, Tensor]:
        edge_index, _ = dropout_adj(self.data.edge_index, p=e_mask_ratio)
        f_mask = torch.empty((self.data.x.shape[1], )).uniform_(0,1) < f_mask_ratio 
        features = self.data.x.clone()
        features[:, f_mask] = 0
        return features, edge_index
    
    def decoder_projection(self, z: Tensor) -> Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def compute_InfoNCE_loss(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        refl_sim = torch.exp(torch.mm(z1, z1.t()) / self.tau) # inter-view
        between_sim = torch.exp(torch.mm(z1, z2.t()) / self.tau) # intra-view

        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )
    
    def make_loss(self, embeddings: Tensor):
        # Generate the two views
        features1, edge_index1 = self.generate_view(self.feature_mask_ratio1, self.edge_mask_ratio1)
        features2, edge_index2 = self.generate_view(self.feature_mask_ratio2, self.edge_mask_ratio2)

        # Compute embeddings of both views
        z1 = self.encoder(features1, edge_index1)
        z2 = self.encoder(features2, edge_index2)

        # Project embeddings via decoder
        h1 = self.decoder_projection(z1)
        h2 = self.decoder_projection(z2)

        # Compute loss
        l1 = self.compute_InfoNCE_loss(h1, h2)
        l2 = self.compute_InfoNCE_loss(h2, h1)
        return ((l1 + l2) * 0.5).mean()


