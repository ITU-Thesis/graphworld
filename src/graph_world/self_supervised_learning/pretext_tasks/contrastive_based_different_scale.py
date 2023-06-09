from torch_geometric.nn.models.deep_graph_infomax import DeepGraphInfomax as DeepGraphInfomaxModule
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import torch
import gin
from .basic_pretext_task import BasicPretextTask
from ..augmentation import node_feature_shuffle
#from torchmetrics.functional import pairwise_cosine_similarity
from typing import Union
from ..loss import jensen_shannon_loss
from ..graph import SubGraph, SubGraphs
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
import math
from .utils import get_exact_ppr_matrix


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

    def __init__(self, k: int, temperature: float, num_iter: int, out_channels: int, **kwargs):
        super().__init__()
        self.k = k
        self.temperature = torch.tensor(temperature)
        self.num_iter = num_iter
        self.out_channels = out_channels

    def compute(self, data: Tensor, num_iter: int, mu_init: Tensor):
        # [0, 1] normalize
        data = data / (data.norm(dim=1)[:, None] + 1e-8)
        mu = mu_init

        for _ in range(num_iter):
            mu = mu / (mu.norm(dim=1, p='fro')[:, None] + 1e-8)

            # Get distances & compute similarities
            dist = torch.mm(data, mu.transpose(0,1))                        # (N observations x N clusters)
            r = F.softmax(self.temperature * dist, dim=1)                        # (N observations x N clusters)
            cluster_r = r.sum(dim=0)                                        # (N clusters)
            cluster_mean = r.t() @ data                                     # (N clusters x N feats)
            new_mu = torch.diag(1 / cluster_r) @ cluster_mean               # (N clusters x N feats)
            mu = new_mu


        return mu, r

    def forward(self, embeddings: Tensor) -> Union[Tensor, Tensor]:
        mu_init = torch.rand(self.k, self.out_channels)
        mu_init, _ = self.compute(data=embeddings, num_iter=self.num_iter, mu_init=mu_init)
        mu, r = self.compute(
            data=embeddings, num_iter=1, mu_init=mu_init.clone().detach())

        return mu, r


@gin.configurable
class GraphInfoClust(BasicPretextTask):
    def __init__(self, cluster_ratio: float, temperature : float, alpha: float,  **kwargs):
        super().__init__(**kwargs)
        assert alpha >= 0. and alpha <= 1.
        k = math.ceil(self.data.x.shape[0]*cluster_ratio)
        def summary_fn(h, *args, **kwargs): return h.mean(dim=0)
        self.alpha = alpha
        self.cluster = ClusterNet(k=k, temperature=temperature, num_iter=10, out_channels=self.encoder.out_channels)
        self.dgi = DeepGraphInfomaxModule(
            hidden_channels=self.encoder.out_channels,
            encoder=self.encoder,
            summary=summary_fn,
            corruption=node_feature_shuffle
        )

    def clustering_discriminator(self, embedding: Tensor, summary: Tensor) -> Tensor:
        '''
        Discriminator for the clustering
        '''
        N, D = embedding.shape
        inner_product_similarity = torch.sigmoid(
            torch.bmm(embedding.view(N, 1, D), summary.view(N, D, 1)).squeeze())
        return inner_product_similarity

    def make_loss(self, embedding: Tensor):
        # DGI (global) objective
        pos_z, neg_z, summary = self.dgi(self.data.x, self.data.edge_index)
        dgi_loss = self.dgi.loss(pos_z=pos_z, neg_z=neg_z, summary=summary)
        # Clustering (coarse-grained) objective
        mu, r = self.cluster(pos_z)
        # (N observations x cluster dim)
        cluster_summary = torch.sigmoid(r @ mu)
        positive_score = self.clustering_discriminator(pos_z, cluster_summary).squeeze()
        negative_score = self.clustering_discriminator(neg_z, cluster_summary).squeeze()

        cluster_loss = jensen_shannon_loss(
            positive_instance=positive_score, negative_instance=negative_score)
        return self.alpha * dgi_loss + (1 - self.alpha) * cluster_loss


@gin.configurable
class SUBGCON(BasicPretextTask):
    '''
    Proposed by:
        Jiao, Yizhu, m.fl. “Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning”. arXiv preprint arXiv:2009.10273, 2020.
    '''

    def __init__(self, alpha: float, k: int, margin: float = 1/2, **kwargs):
        super().__init__(**kwargs)
        assert alpha >= 0. and alpha <= 1.
        assert k > 0

        self.N = self.data.num_nodes

        S = get_exact_ppr_matrix(data=self.data, alpha=alpha)
        S = S.fill_diagonal_(S.min() - 1)


        # Take the k most important neighbours
        # S_top_k = S.topk(k=k, dim=1).indices
        # S_top_k = torch.cat([
        #     S_top_k, 
        #     torch.arange(start=0, end=self.data.num_nodes, step=1).unsqueeze(dim=1)
        # ], dim=1)
        top_k = []
        S_top_k = S.topk(k=k, dim=1)
        for target, (vals, idx) in enumerate(zip(*S_top_k)):
            non_zero_mask = vals != 0
            subgraph_nodes = idx[non_zero_mask].detach().tolist() + [target]
            top_k += [subgraph_nodes]

        self.loss = torch.nn.MarginRankingLoss(margin=margin, reduction='mean')

        # Subgraphs for each node
        ss = []
        for node_indices in top_k:
            ss += [SubGraph(node_indices=node_indices, data=self.data)]
        self.subgraphs = SubGraphs(ss)


        # Used for the picking function
        self.central_node_indices = [None] * self.subgraphs.n_subgraphs
        
        for i in range(self.subgraphs.n_subgraphs):
            self.central_node_indices[i] = self.subgraphs.get_subgraph_offset(i) +\
                self.subgraphs.get_subgraph(i).node_mapping.src_to_target(i)
 

    def __get_embedding_and_summaries(self) -> Union[Tensor, Tensor]:
        all_embeddings = self.encoder(
            self.subgraphs.subgraph_batches.x, self.subgraphs.subgraph_batches.edge_index)
        summaries = torch.sigmoid(global_mean_pool(
            x=all_embeddings, batch=self.subgraphs.subgraph_batches.batch))
        # Picking function
        embeddings = all_embeddings[self.central_node_indices, :]

        return embeddings, summaries

    def get_downstream_embeddings(self) -> Tensor:
        return self.__get_embedding_and_summaries()[0]

    def make_loss(self, embeddings, **kwargs):
        rand_idx = torch.randperm(self.N)
        embeddings1, summaries1 = self.__get_embedding_and_summaries()

        summaries2 = summaries1[rand_idx]
        embeddings2 = embeddings1[rand_idx]

        positives1 = torch.sigmoid((embeddings1 * summaries1).sum(dim=1))
        negatives1 = torch.sigmoid((embeddings1 * summaries2).sum(dim=1))

        positives2 = torch.sigmoid((embeddings2 * summaries2).sum(dim=1))
        negatives2 = torch.sigmoid((embeddings2 * summaries1).sum(dim=1))

        ones = torch.ones(self.N)

        loss_1 = self.loss(positives1, negatives1, ones)
        loss_2 = self.loss(positives2, negatives2, ones)

        return loss_1 + loss_2

