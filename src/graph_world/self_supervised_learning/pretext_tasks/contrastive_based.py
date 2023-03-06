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
from torch_geometric.utils import negative_sampling, dropout_adj, degree
from ..layers import NeuralTensorLayer
from typing import Tuple
import copy
from .utils import EMA, init_weights

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
    

# Based on https://github.com/CRIPAC-DIG/GCA
# We only use node degree as centrality
# - Can also use pagerank or eigenvector centrality
@gin.configurable
class GCA(GRACE):
    def __init__(self, 
                 tau : float = 0.5, 
                 edge_mask_ratio1 : float = 0.2,
                 edge_mask_ratio2 : float = 0.2,
                 feature_mask_ratio1 : float = 0.2, 
                 feature_mask_ratio2 : float = 0.2,
                 **kwargs):
        super().__init__(tau, edge_mask_ratio1, edge_mask_ratio2, 
                         feature_mask_ratio1, feature_mask_ratio2, **kwargs)
        
        node_degrees = degree(self.data.edge_index[1])

        # Edge masking weights
        deg_col = node_degrees[self.data.edge_index[1]].to(torch.float32)
        s_col = torch.log(deg_col)
        self.edge_weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())


        # Feature masking weights (for dense continous features)
        # - Should be modified if features are discrete and sparse
        x = self.data.x.abs()
        w = x.t() @ node_degrees
        w = w.log()
        self.feature_weights = (w.max() - w) / (w.max() - w.mean())

    # Drops edges based on edge_weights and mask_ratio
    def drop_edges(self, e_mask_ratio : float, threshold : float = 0.7) -> Tensor:
        edge_weights = self.edge_weights / self.edge_weights.mean() * e_mask_ratio
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        return self.data.edge_index[:, mask]
    
    # Drop features based on feature_weights and mask_ratio
    def drop_features(self, f_mask_ratio: float, threshold:float = 0.7) -> Tensor:
        w = self.feature_weights / self.feature_weights.mean() * f_mask_ratio
        w = w.where(w < threshold, torch.ones_like(w) * threshold)
        mask = torch.bernoulli(w).to(torch.bool)
        features = self.data.x.clone()
        features[:, mask] = 0
        return features

    # Override augmentation to take weights into account
    def generate_view(self, f_mask_ratio : float, e_mask_ratio : float) -> Tuple[Tensor, Tensor]:
        # Use threshold=0.7 similar to authors of method
        edge_index = self.drop_edges(e_mask_ratio=e_mask_ratio, threshold=0.7) 
        features = self.drop_features(f_mask_ratio=f_mask_ratio, threshold=0.7)
        return features, edge_index


# Based on https://github.com/Namkyeong/BGRL_Pytorch
@gin.configurable
class BGRL(BasicPretextTask):
    def __init__(self, 
                 edge_mask_ratio1 : float = 0.2,
                 edge_mask_ratio2 : float = 0.2,
                 feature_mask_ratio1 : float = 0.2, 
                 feature_mask_ratio2 : float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Augmentation params
        self.edge_mask_ratio1 = edge_mask_ratio1
        self.edge_mask_ratio2 = edge_mask_ratio2
        self.feature_mask_ratio1 = feature_mask_ratio1
        self.feature_mask_ratio2 = feature_mask_ratio2

        # Create student and teacher encoder
        # No gradients are needed for teacher as EMA is used
        # We also initialize the teacher with different weights than the student
        self.student_encoder = self.encoder
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_ema_updater = EMA(0.99, self.epochs) # Fix initial decay to 0.99 similar to authors
        self.teacher_encoder.apply(init_weights)

        # Create predictor for student -> teacher
        out = self.encoder.out_channels
        self.student_predictor = torch.nn.Sequential(
            Linear(out, out), torch.nn.PReLU(),
            Linear(out, out)
        )

    # Same as in GRACE
    def generate_view(self, f_mask_ratio : float, e_mask_ratio : float) -> Tuple[Tensor, Tensor]:
        edge_index, _ = dropout_adj(self.data.edge_index, p=e_mask_ratio)
        f_mask = torch.empty((self.data.x.shape[1], )).uniform_(0,1) < f_mask_ratio 
        features = self.data.x.clone()
        features[:, f_mask] = 0
        return features, edge_index
    
    # BOYE loss
    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def make_loss(self, embeddings: Tensor):
        # Generate two views
        features1, edge_index1 = self.generate_view(self.feature_mask_ratio1, self.edge_mask_ratio1)
        features2, edge_index2 = self.generate_view(self.feature_mask_ratio2, self.edge_mask_ratio2)

        # Produce student embeddings
        v1_student = self.student_encoder(features1, edge_index1)
        v2_student = self.student_encoder(features2, edge_index2)

        # Produce teacher embeddings
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(features1, edge_index1)
            v2_teacher = self.teacher_encoder(features2, edge_index2)

        # Predict teacher embeddings from student embeddings
        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)

        # Update teacher encoder params
        # We update the teacher before the student rather than after
        # - This fits our interface better, as the benchmarker updates the student
        # - The order should not matter
        for current_params, ma_params in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.teacher_ema_updater.update_average(old_weight, up_weight)

        # Compute loss for predictions
        # The returned loss updates the student encoder
        # Note that predictions from view 1 are for view 2 and vice versa
        loss1 = self.loss_fn(v1_pred, v2_teacher.detach())
        loss2 = self.loss_fn(v2_pred, v1_teacher.detach())
        loss = loss1 + loss2
        return loss.mean()


        