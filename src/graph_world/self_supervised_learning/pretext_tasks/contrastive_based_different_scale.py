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
from torch_ppr import personalized_page_rank
from ..graph import Subgraph
from torch_geometric.nn import global_mean_pool


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

@gin.configurable
class SUBG_CON(BasicPretextTask):
    '''
    Proposed by:
        Jiao, Yizhu, m.fl. “Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning”. arXiv preprint arXiv:2009.10273, 2020.
    '''
    def __init__(self, alpha : float, k: int, margin : float = 1/2, **kwargs):
        super().__init__(**kwargs)
        assert alpha >= 0. and alpha <= 1.
        assert k > 0

        self.N = self.data.num_nodes
        S = personalized_page_rank(edge_index=self.data.edge_index, alpha=alpha)
        S_top_k = S.topk(k=k, dim=1).indices

        self.loss = torch.nn.MarginRankingLoss(margin=margin, reduction='mean')

        # Subgraphs for each node
        self.subgraphs = [Subgraph(node_indices=S_top_k[i, :], data=self.data) for i in range(self.N)]


    def __get_embedding_and_summary(self) -> Union[Tensor, Tensor]:
        embeddings = [None] * self.N
        summaries = [None] * self.N

        for i in range(self.N):
            subgraph_i = self.subgraphs[i]
            subgraph_data = subgraph_i.subgraph_data
            i_new_index = subgraph_i.get_old_to_new_index(i)

            H_i = self.encoder(subgraph_data.x, subgraph_data.edge_index)
            embeddings[i] = H_i[i_new_index, :][None, :]
            summaries[i] = self.__readout(H_i)

        embeddings, summaries = torch.concat(embeddings).squeeze(), torch.concat(summaries)
        
        assert embeddings.shape == summaries.shape
        return embeddings, summaries
    
    def get_downstream_embeddings(self) -> Tensor:
        return self.__get_embedding_and_summary()[0]
        
    def __readout(self, H_i : Tensor):
        return torch.sigmoid(global_mean_pool(x=H_i, batch=None))
    

    def make_loss(self, embeddings, **kwargs):
        rand_idx = torch.randperm(self.N)
        embeddings1, summaries1 = self.__get_embedding_and_summary()
        
        summaries2 = summaries1[rand_idx]
        embeddings2 = summaries1[rand_idx]

        positives1 = torch.sigmoid((embeddings1 * summaries1).sum(dim=1))
        negatives1 = torch.sigmoid((embeddings1 * summaries2).sum(dim=1))

        positives2 = torch.sigmoid((embeddings2 * summaries2).sum(dim=1))
        negatives2 = torch.sigmoid((embeddings2 * summaries1).sum(dim=1))

        ones = torch.ones(self.N)

        loss_1 = self.loss(positives1, negatives1, ones)
        loss_2 = self.loss(positives2, negatives2, ones)

        return loss_1 + loss_2

