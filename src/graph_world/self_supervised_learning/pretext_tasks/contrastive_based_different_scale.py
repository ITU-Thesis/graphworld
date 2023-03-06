from torch_geometric.nn.models.deep_graph_infomax import DeepGraphInfomax as DeepGraphInfomaxModule
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import torch
import gin
from .basic_pretext_task import BasicPretextTask
from ..augmentation import node_feature_shuffle
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Union
from ..loss import jensen_shannon_loss

@gin.configurable
class DeepGraphInfomax(BasicPretextTask):
    '''
    Deep Graph Infomax proposed in Velickovic, Petar, et al. "Deep graph infomax." ICLR (Poster) 2.3 (2019): 4.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        def summary_fn(h, *args, **kwargs): return h.mean(dim=0)

        self.dgi = DeepGraphInfomaxModule(
            hidden_channels=self.encoder.out_channels,
            encoder=self.encoder,
            summary=summary_fn,
            corruption=node_feature_shuffle
        )

    def make_loss(self, embeddings: Tensor):
        pos_z, neg_z, summary = self.dgi(self.data.x, self.data.edge_index)
        return self.dgi.loss(pos_z=pos_z, neg_z=neg_z, summary=summary)


class ClusterNet(Module):
    '''
    ClusterNet propoesd in Wilder, B., E. Ewing, B. Dilkina, and M. Tambe (2019). End to end learning and optimization on graphs.
    Implementation modified from: https://github.com/cmavro/Graph-InfoClust-GIC
    Original implementation of ClusterNet can be found here: https://github.com/bwilder0/clusternet
    '''
    def __init__(self, k: int, temperature : float, num_iter : int, out_channels : int, **kwargs):
        super().__init__()
        assert temperature >= 0. and temperature <= 1.
        self.k = k
        self.temperature = temperature
        self.num_iter = num_iter
        self.out_channels = out_channels

    def compute(self, data : Tensor, num_iter : int, mu_init : Tensor):
        # [0, 1] normalize
        data = data / (data.norm(dim=1) + 1e-8)[:, None]

        mu = mu_init

        for _ in range(num_iter):
            mu = mu / mu.norm(dim=1, p='fro')[:, None]

            # Get distances & compute cosine similarity
            cosine_similarities = pairwise_cosine_similarity(data, mu)      # (N observations x N clusters)
            r = F.softmax(-self.temperature * cosine_similarities, dim=1)   # (N observations x N clusters)
            cluster_r = r.sum(dim=0)                                        # (N clusters)
            cluster_mean = r.t() @ data                                     # (N clusters x N feats)
            new_mu = torch.diag(1 / cluster_r) @ cluster_mean               # (N clusters x N feats)
            mu = new_mu

        return mu, r

    def forward(self, embeddings : Tensor) -> Union[Tensor, Tensor]:
        mu_init = torch.rand(self.k, self.out_channels)
        mu_init, _ = self.compute(data=embeddings, num_iter=1, mu_init=mu_init)
        mu, r = self.compute(data=embeddings, num_iter=self.num_iter, mu_init=mu_init)
        
        return mu, r

        
@gin.configurable
class GraphInfoClust(BasicPretextTask):
    def __init__(self, k: int, temperature : float, num_cluster_iter : int, alpha: float,  **kwargs):
        super().__init__(**kwargs)
        assert alpha >= 0. and alpha <= 1.
        def summary_fn(h, *args, **kwargs): return h.mean(dim=0)
        self.alpha = alpha
        self.cluster = ClusterNet(k=k, temperature=temperature, num_iter=num_cluster_iter, out_channels=self.encoder.out_channels)
        self.dgi = DeepGraphInfomaxModule(
            hidden_channels=self.encoder.out_channels,
            encoder=self.encoder,
            summary=summary_fn,
            corruption=node_feature_shuffle
        )
        

    def clustering_discriminator(self, embedding : Tensor, summary : Tensor) -> Tensor:
        '''
        Discriminator for the clustering
        '''
        N, D = embedding.shape
        inner_product_similarity = torch.sigmoid(torch.bmm(embedding.view(N, 1, D), summary.view(N, D, 1)).squeeze())
        return inner_product_similarity


    def make_loss(self, embedding : Tensor):
        # DGI (global) objective
        pos_z, neg_z, summary = self.dgi(self.data.x, self.data.edge_index)
        dgi_loss = self.dgi.loss(pos_z=pos_z, neg_z=neg_z, summary=summary)

        # Clustering (coarse-grained) objective
        mu, r = self.cluster(pos_z)
        cluster_summary = torch.sigmoid(r @ mu) # (N observations x cluster dim)
        positive_score = self.clustering_discriminator(pos_z, cluster_summary).squeeze()
        negative_score = self.clustering_discriminator(neg_z, cluster_summary).squeeze()
        cluster_loss = jensen_shannon_loss(positive_instance=positive_score, negative_instance=negative_score)

        return self.alpha * dgi_loss + (1 - self.alpha) * cluster_loss       
