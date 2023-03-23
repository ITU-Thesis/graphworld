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
from torch_geometric.utils import negative_sampling, dropout_adj, degree, subgraph
from .pyg_compatability_utils import add_random_edge
from ..layers import NeuralTensorLayer
from typing import Tuple
import copy
from .utils import EMA, init_weights, pad_views, compute_InfoNCE_loss
from abc import ABC, abstractclassmethod
from torch_geometric.transforms import GDC, LocalDegreeProfile

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

    def generate_view(self, f_mask_ratio : float, e_mask_ratio : float) -> Tuple[Tensor, Tensor]:
        edge_index, _ = dropout_adj(self.data.edge_index, p=e_mask_ratio)
        f_mask = torch.empty((self.data.x.shape[1], )).uniform_(0,1) < f_mask_ratio 
        features = self.data.x.clone()
        features[:, f_mask] = 0
        return features, edge_index
    
    def decoder_projection(self, z: Tensor) -> Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
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
        l1 = compute_InfoNCE_loss(h1, h2, self.tau)
        l2 = compute_InfoNCE_loss(h2, h1, self.tau)
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
        
        node_degrees = degree(self.data.edge_index[1], num_nodes=self.data.num_nodes)

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
# and https://github.com/zekarias-tilahun/SelfGNN
# This class implements the siamese architecture avoiding negative samples
class AbstractSiameseBYOL(BasicPretextTask, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create student and teacher encoder
        # No gradients are needed for teacher as EMA is used
        # We also initialize the teacher with different weights than the student
        self.student_encoder = self.encoder
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_ema_updater = EMA(0.99, self.epochs) # Fix initial decay to 0.99
        self.teacher_encoder.apply(init_weights)

        # Create predictor for student -> teacher
        out = self.encoder.out_channels
        self.student_predictor = torch.nn.Sequential(
            Linear(out, out), torch.nn.PReLU(),
            Linear(out, out)
        )

    # Override this function to generate 2 views.
    # should return tuple of (features1, edge_index1, features2, edge_index2)
    @abstractclassmethod
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    # Compute loss for predictions
    # The returned loss updates the student encoder
    # Note that predictions from view 1 are for view 2 and vice versa
    def compute_loss(self, v1_teacher, v2_teacher, v1_pred, v2_pred):
        loss1 = self.loss_fn(v1_pred, v2_teacher.detach())
        loss2 = self.loss_fn(v2_pred, v1_teacher.detach())
        loss = loss1 + loss2
        return loss.mean()

    def make_loss(self, embeddings: Tensor):
        # @TODO Bertram, fix edge_weight interface everywhere
        # - Some methods use it, set default to 1 and let them override it.
        # - Also make generate_views return data object instead
        # Generate two views
        features1, edge_index1, edge_weights1, features2, edge_index2, edge_weights2 = self.generate_views()

        # Produce student embeddings
        v1_student = self.student_encoder(features1, edge_index1, edge_weights1)
        v2_student = self.student_encoder(features2, edge_index2, edge_weights2)

        # Produce teacher embeddings
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(features1, edge_index1, edge_weights1)
            v2_teacher = self.teacher_encoder(features2, edge_index2, edge_weights2)

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

        return self.compute_loss(v1_teacher, v2_teacher, v1_pred, v2_pred)

@gin.configurable
class BGRL(AbstractSiameseBYOL):
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

    # EM an NFM
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_index1, _ = dropout_adj(self.data.edge_index, p=self.edge_mask_ratio1)
        edge_index2, _ = dropout_adj(self.data.edge_index, p=self.edge_mask_ratio2)

        f_mask1 = torch.empty((self.data.x.shape[1], )).uniform_(0,1) < self.feature_mask_ratio1 
        f_mask2 = torch.empty((self.data.x.shape[1], )).uniform_(0,1) < self.feature_mask_ratio2 

        features1 = self.data.x.clone()
        features2 = self.data.x.clone()

        features1[:, f_mask1] = 0
        features2[:, f_mask2] = 0
        return features1, edge_index1, None, features2, edge_index2, None
    
# Abstract class - does not override generate_views
# This class modifies the expected embeddings used by the graph encoder
# Extensions of this class are the augmentation variatns of SelfGNN 
class SelfGNN(AbstractSiameseBYOL):
    # Concat embeddings of the two views
    def get_downstream_embeddings(self) -> Tensor:
        # Generate two views
        features1, edge_index1, edge_weights1, features2, edge_index2, edge_weights2 = self.generate_views()

        # Produce student embeddings
        v1_student = self.student_encoder(features1, edge_index1, edge_weights1)
        v2_student = self.student_encoder(features2, edge_index2, edge_weights2)
        return torch.cat([v1_student, v2_student], dim=1)#.detach()

    # Because of concat the output dim might change
    def get_embedding_dim(self) -> int:
        return self.encoder.out_channels * 2
    

# SelfGNN variant that uses node feature split
@gin.configurable
class SelfGNNSplit(SelfGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generate cached views once
        # We randomly split features in two parts
        # As a simplification we mask splitted features as 0 rather than removing them
        # - Makes it so the encoder input dim does not have to change
        perm = torch.randperm(self.data.x.shape[1])
        x = self.data.x.clone()
        x = x[:, perm]
        size = x.shape[1] // 2
        self.features1 = x.clone()
        self.features2 = x.clone()
        self.features1[:, :size] = 0
        self.features2[:, size:] = 0
    
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.features1, self.data.edge_index, None, self.features2, self.data.edge_index, None
    

# SelfGNN variant that uses PPR
@gin.configurable
class SelfGNNPPR(SelfGNN):
    def __init__(self, alpha : float = 0.15, **kwargs):
        super().__init__(**kwargs)
        # Generate cached views once
        # We keep one view as the original graph, and augment the other based on sparsified PPR edges
        # The paper fixes alpha, we vary it as a hyperparameter
        
        # GDC assumes edge_attr stores edge weights
        # To avoid issues where this is not the case from the generated graphs we set the edge_attr to None
        data2 = self.data.clone()
        data2.edge_attr = None
        self.data2 = GDC(diffusion_kwargs={'alpha': alpha, 'method': 'ppr'}, 
                       sparsification_kwargs={'method':'threshold', 'avg_degree': 30})(data2)
    
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.data.x, self.data.edge_index, None, self.data2.x, self.data2.edge_index, self.data2.edge_attr
    


# SelfGNN variant that replaces the features of 1 view with LocalDegreeProfile
@gin.configurable
class SelfGNNLDP(SelfGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generate cached views once
        # We keep one view as the original graph, and augment the other to be LDP of dim=5
        # - feaure dims need to match, so LDP features are padded with 0
        data2 = self.data.clone()
        data2.x = None
        data2.num_nodes = self.data.num_nodes
        self.data2 = LocalDegreeProfile()(data2)
        pad_views(self.data, self.data2)
    
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.data.x, self.data.edge_index, None, self.data2.x, self.data2.edge_index, None
    

# SelfGNN variant that zscore standardizes the features of 1 view
@gin.configurable
class SelfGNNStandard(SelfGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generate cached views once
        # We keep one view as the original graph, and augment the other be a zscore standardization
        self.data2 = self.data.clone()
        x = self.data2.x
        mean, std = x.mean(dim=0), x.std(dim=0)
        self.data2.x = (x - mean) / (std + 10e-7)
    
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.data.x, self.data.edge_index, None, self.data2.x, self.data2.edge_index, None
    


# The G-BT method that uses Barlow Twins loss
# Based on https://github.com/pbielak/graph-barlow-twins
@gin.configurable
class GBT(BasicPretextTask):
    def __init__(self, 
                 edge_mask_ratio : float = 0.2,
                 feature_mask_ratio : float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.edge_mask_ratio =  edge_mask_ratio
        self.feature_mask_ratio = feature_mask_ratio
   
   # EM and NFM (same as in GRACE)
    def generate_view(self, f_mask_ratio : float, e_mask_ratio : float) -> Tuple[Tensor, Tensor]:
        edge_index, _ = dropout_adj(self.data.edge_index, p=e_mask_ratio)
        f_mask = torch.empty((self.data.x.shape[1], )).uniform_(0,1) < f_mask_ratio 
        features = self.data.x.clone()
        features[:, f_mask] = 0
        return features, edge_index
    
    def barlow_twins_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        EPS = 1e-15
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        _lambda = 1 / feature_dim

        # Apply batch normalization
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + EPS)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + EPS)

        # Cross-correlation matrix
        c = (z1_norm.T @ z2_norm) / batch_size

        # Loss function
        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (
            (1 - c.diagonal()).pow(2).sum()
            + _lambda * c[off_diagonal_mask].pow(2).sum()
        )

        return loss
    
    def make_loss(self, embeddings: Tensor):
        # Generate the two views (same masking ratios)
        features1, edge_index1 = self.generate_view(self.feature_mask_ratio, self.edge_mask_ratio)
        features2, edge_index2 = self.generate_view(self.feature_mask_ratio, self.edge_mask_ratio)

        # Compute embeddings of both views
        z1 = self.encoder(features1, edge_index1)
        z2 = self.encoder(features2, edge_index2)

        # Compute Barlow Twins loss
        return self.barlow_twins_loss(z1, z2)

    
@gin.configurable
class MERIT(AbstractSiameseBYOL):
    def __init__(self, alpha : float = 0.15, 
                 sample_size : int = 2000,
                 edge_modification_ratio : float = 0.2,
                 feature_mask_ratio : float = 0.2,
                 beta : float = 0.6,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size
        self.edge_modification_ratio = edge_modification_ratio
        self.feature_mask_ratio = feature_mask_ratio
        self.beta = beta
        # Produce PPR adjacency matrix where edges with weights less than 0.01 are removed 
        # GDC assumes edge_attr stores edge weights
        # - To avoid issues where this is not the case from the generated graphs we set the edge_attr to None
        data_ppr = self.data.clone()
        data_ppr.edge_attr = None
        self.data_ppr = GDC(diffusion_kwargs={'alpha': alpha, 'method': 'ppr'}, 
                sparsification_kwargs={'method':'threshold', 'eps': 0.01})(data_ppr)
    
    def mask_features(self, features):
        f_mask = torch.empty((features.shape[1], )).uniform_(0,1) < self.feature_mask_ratio
        f = features.clone()
        f[:, f_mask] = 0
        return f
    
    def sub_sample(self, sampled_nodes, data) -> Tuple[Tensor, Tensor]:
        return subgraph(sampled_nodes, data.edge_index, data.edge_attr, 
                                        relabel_nodes = True, num_nodes = data.num_nodes)
    
    def create_views(self, sub_sample : bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if sub_sample and self.sample_size < self.data.num_nodes:
            # Nodes to be used for subsampling
            all_nodes = np.arange(self.data.x.shape[0])
            perm = np.random.permutation(all_nodes)
            sampled_nodes = perm[: self.sample_size]
            sampled_nodes = sorted(sampled_nodes) # Need because of how labels are handled when subsampling
            edge_index1, _ = subgraph(sampled_nodes, self.data.edge_index, 
                                              relabel_nodes = True, num_nodes = self.data.num_nodes)
            edge_index2, edge_weights2 = subgraph(sampled_nodes, self.data_ppr.edge_index, self.data_ppr.edge_attr, 
                                        relabel_nodes = True, num_nodes = self.data_ppr.num_nodes)
            features = self.data.x[sampled_nodes,:]
        else:
            edge_index1 = self.data.edge_index
            edge_index2, edge_weights2 = self.data_ppr.edge_index, self.data_ppr.edge_attr
            features = self.data.x
        
        # Generate augmentation 1: (SS) + EM + NFM
        _, new_edges = add_random_edge(edge_index1, p=self.edge_modification_ratio / 2, force_undirected = True)
        edge_index1, _ = dropout_adj(edge_index1, p=self.edge_modification_ratio / 2)
        edge_index1 = torch.cat([edge_index1, new_edges], dim=1)
        features1 = self.mask_features(features)

        # Generate augmentation 2: (SS) + PPR + NFM
        features2 = self.mask_features(features)

        return features1, edge_index1, None, features2, edge_index2, edge_weights2
    
    # Override parent class
    def generate_views(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.create_views(sub_sample = True)

    def contrastive_loss_cross_network(self, h1, z):
        z1 = F.normalize(h1)
        z2 = F.normalize(z)
        cross_sim = torch.exp(torch.mm(z1, z2.t()))
        return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

    # Override loss function 
    def compute_loss(self, v1_teacher, v2_teacher, v1_pred, v2_pred):
        l1 = self.beta * compute_InfoNCE_loss(v1_pred, v2_pred) + \
            (1.0 - self.beta) * self.contrastive_loss_cross_network(v1_pred, v2_teacher.detach())
        
        l2 = self.beta * compute_InfoNCE_loss(v2_pred, v1_pred) + \
            (1.0 - self.beta) * self.contrastive_loss_cross_network(v2_pred, v1_teacher.detach())
        
        loss = 0.5 * (l1 + l2)
            
        return loss.mean()
    
    # Add embeddings of the two views
    def get_downstream_embeddings(self) -> Tensor:
        # Generate two views
        features1, edge_index1, edge_weights1, features2, edge_index2, edge_weights2 = self.create_views(sub_sample = False)

        # Produce student embeddings
        v1_student = self.student_encoder(features1, edge_index1, edge_weights1)
        v2_student = self.student_encoder(features2, edge_index2, edge_weights2)
        return v1_student + v2_student






        