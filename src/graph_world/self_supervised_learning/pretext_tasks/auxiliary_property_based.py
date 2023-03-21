from torch.nn import Linear
import numpy as np
import torch
import gin
from .__types import *
from torch import Tensor
from sklearn.cluster import KMeans
import pymetis
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from .basic_pretext_task import BasicPretextTask
from enum import Enum
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from abc import ABC
from torch import nn
from typing import List, Tuple
from torch_geometric.utils.undirected import is_undirected
from networkx import all_pairs_shortest_path_length
from ..tensor_utils import get_top_k_indices
from torchmetrics.functional import pairwise_cosine_similarity
import math 
import random

# ==================================================== #
# ============= Auxiliary property-based ============= #
# ==================================================== #


@gin.configurable
class NodeClusteringWithAlignment(BasicPretextTask):
    '''
    Paper from:
        You, Yuning, et al. "When does self-supervision help graph convolutional networks?." international conference on machine learning. PMLR, 2020.
    Implementation modified from https://github.com/Shen-Lab/SS-GCNs/blob/master/SS-GCNs/clu.py
    '''
    def __init__(self, cluster_ratio: float, **kwargs):
        super().__init__(**kwargs)

        n_clusters = math.ceil(self.data.x.shape[0]*cluster_ratio)

        # Step 0: Setup
        X, y = self.data.x, self.data.y
        num_classes = y.unique().shape[0]
        feat_dim = X.shape[1]
        centroids_labeled = torch.zeros((num_classes, feat_dim))

        # Step 1: Compute centroids in each cluster by the mean in each class
        for cn in range(num_classes):
            lf = X[self.train_mask]
            ll = y[self.train_mask]
            centroids_labeled[cn] = lf[ll == cn].mean(axis=0)

        # Step 2: Set cluster labels for each node
        cluster_labels = torch.ones(y.shape, dtype=torch.int64) * -1
        cluster_labels[self.train_mask] = y[self.train_mask]

        # Step 3: Train KMeans on all points
        kmeans = KMeans(n_clusters=n_clusters).fit(X)

        # Step 4: Perform alignment mechanism
        #   1) Compute its centroids
        #   2) Find cluster closest to the centroid computed in step 1
        #   3) Assign all unlabeled nodes to that closest cluster.
        for cn in range(n_clusters):
            # v_l - Note we exclude the training data as this is only for the unlabeled data.
            centroids_unlabeled = X[torch.logical_and(torch.tensor(kmeans.labels_ == cn), ~self.train_mask)].mean(axis=0)

            # Equation 5
            label_for_cluster = np.linalg.norm(centroids_labeled - centroids_unlabeled, axis=1).argmin()
            for node in np.where(kmeans.labels_ == cn)[0]:
                if not self.train_mask[node]:
                    cluster_labels[node] = label_for_cluster

        self.pseudo_labels = cluster_labels
        self.decoder = Linear(self.encoder.out_channels, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def make_loss(self, embeddings):
        y_hat = self.decoder(embeddings)
        return self.loss(input=y_hat[~self.train_mask], target=self.pseudo_labels[~self.train_mask])


@gin.configurable
class GraphPartitioning(BasicPretextTask):
    def __init__(self, n_partitions: int, **kwargs):
        super().__init__(**kwargs)

        sparse_matrix = to_scipy_sparse_matrix(self.data.edge_index)
        node_num = sparse_matrix.shape[0]
        adj_list = [[] for _ in range(node_num)]
        for i, j in zip(sparse_matrix.row, sparse_matrix.col):
            if i == j:
                continue
            adj_list[i].append(j)

        _, ss_labels = pymetis.part_graph(adjacency=adj_list, nparts=n_partitions)

        self.pseudo_labels = torch.tensor(ss_labels, dtype=torch.int64)
        self.decoder = Linear(self.encoder.out_channels, n_partitions)
        self.loss = torch.nn.CrossEntropyLoss()

    def make_loss(self, embeddings):
        y_hat = self.decoder(embeddings)
        return self.loss(input=y_hat, target=self.pseudo_labels)


class CentralityScore_(Enum):
    EIGENVECTOR_CENTRALITY = 0
    BETWEENNESS = 1
    CLOSENESS = 2
    SUBGRAPH = 3

class AbstractCentralityScore(BasicPretextTask, ABC):
    '''
    Abstract class for computing centrality scores proposed in https://arxiv.org/pdf/1905.13728.pdf.
    '''
    def __init__(self, centrality_score: CentralityScore_, **kwargs):
        super().__init__(**kwargs)
        assert is_undirected(edge_index=self.data.edge_index)
        if centrality_score == CentralityScore_.EIGENVECTOR_CENTRALITY:
            self.centrality_score_fn = nx.eigenvector_centrality
        elif centrality_score == CentralityScore_.BETWEENNESS:
            self.centrality_score_fn = nx.betweenness_centrality
        elif centrality_score == CentralityScore_.CLOSENESS:
            self.centrality_score_fn = nx.closeness_centrality
        elif centrality_score == CentralityScore_.SUBGRAPH:
            self.centrality_score_fn = nx.subgraph_centrality
        else:
            raise 'Unknown centrality score.'

        layers = [self.encoder.out_channels, max(
            self.encoder.out_channels, 1), 1]

        self.decoder = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Linear(layers[1], layers[2])
        )


        G = to_networkx(self.data).to_undirected()
        centralities = self.centrality_score_fn(G)
        scores = torch.tensor([centralities[i]
                                for i in range(len(centralities))])
        rank_order = torch.zeros((scores.shape[0], scores.shape[0]))
        for i in range(len(scores)):
            for j in range(len(scores)):
                if scores[i] > scores[j]:
                    rank_order[i, j] = 1.0
        self.rank_order = rank_order

    def make_loss(self, embeddings):
        predicted_centrality_score = self.decoder(embeddings)
        
        # Outer subtraction followed by element-wise sigmoid
        predicted_rank_order = torch.sigmoid(
            predicted_centrality_score - predicted_centrality_score.T
        )
        R, R_hat = self.rank_order, predicted_rank_order
        loss = -(torch.log(R * R_hat + 1e-8) + (1 - R) * torch.log((1 - R_hat) + 1e-8)).mean() # Elementwise CE loss followed by mean
        return loss

@gin.configurable
class EigenvectorCentrality(AbstractCentralityScore):
    '''
    Proposed in https://arxiv.org/pdf/1905.13728.pdf.
    Time complexity: O(|V|^3) where V is the vertices.
    '''
    def __init__(self, **kwargs):
        super().__init__(CentralityScore_.EIGENVECTOR_CENTRALITY, **kwargs)


@gin.configurable
class BetweennessCentrality(AbstractCentralityScore):
    '''
    Proposed in https://arxiv.org/pdf/1905.13728.pdf.
    Time complexity: O(|V| * |E|) where V and E is the vertices and edges respectively.
    '''
    def __init__(self, **kwargs):
        super().__init__(CentralityScore_.BETWEENNESS, **kwargs)


@gin.configurable
class ClosenessCentrality(AbstractCentralityScore):
    '''
    Proposed in https://arxiv.org/pdf/1905.13728.pdf.
    Time complexity: O(|V| * |E|) where V and E is the vertices and edges respectively.
    '''
    def __init__(self, **kwargs):
        super().__init__(CentralityScore_.CLOSENESS, **kwargs)


@gin.configurable
class SubgraphCentrality(AbstractCentralityScore):
    '''
    Proposed in https://arxiv.org/pdf/1905.13728.pdf.
    Time complexity: O(|V|^4) where V is the vertices.
    '''
    def __init__(self, **kwargs):
        super().__init__(CentralityScore_.SUBGRAPH, **kwargs)


@gin.configurable
class CentralityScoreRanking(BasicPretextTask):
    '''
    Proposed in https://arxiv.org/pdf/1905.13728.pdf.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.centrality_scores = nn.ModuleList([
            EigenvectorCentrality(**kwargs),
            BetweennessCentrality(**kwargs),
            ClosenessCentrality(**kwargs),
            SubgraphCentrality(**kwargs)
        ])

    def make_loss(self, embeddings):
        loss = sum(map(lambda m: m.make_loss(embeddings), self.centrality_scores)) * 0.25
        return loss

@gin.configurable
class S2GRL(BasicPretextTask):
    '''
    Implementation of S2GRL from:
        "Peng, Zhen, et al. "Self-supervised graph representation learning via global context prediction."

    It implements the "small-world" merge policy as used in their experiments described in table 6.
    According to the paper reduction is always sum, but this might make the model improve more towards
    hubs / nodes in highly dense neighbourhoods.
    '''
    def __init__(self, shortest_path_classes : Tuple[int,int], sample_size : float, **kwargs):
        super().__init__(**kwargs)
        shortest_path_cutoff, N_classes = shortest_path_classes

        assert shortest_path_cutoff > 0, 'cutoff shout be greater than 0'
        assert sample_size > 0. and sample_size <= 1., f'sample size of value {sample_size} must be bewtween 0 and 1.'

        self.N_classes = N_classes
        self.N_sampled_nodes = int(sample_size * self.data.num_nodes)
        in_channel = self.encoder.out_channels

        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.decoder = nn.Sequential(
                Linear(in_channel, in_channel // 2),
                nn.ReLU(),
                Linear(in_channel // 2, self.N_classes),
        )        
        

        self.k_hop_neighbors_source_indices = [None] * self.data.num_nodes      # Src node --> Src node
        self.k_hop_neighbors_target_indices = [None] * self.data.num_nodes      # Src node --> indices of K-hop neighbors
        self.k_hop_neighbors_distances = [None] * self.data.num_nodes           # Src node --> distance of K-hop neighbors
        
        G = to_networkx(self.data)
        self.indices = [*range(0, self.data.num_nodes)]
        shortest_paths = all_pairs_shortest_path_length(G, cutoff=shortest_path_cutoff)
        shortest_paths = map(lambda v_i: (v_i[0], filter(lambda neighbours: neighbours[1] > 0, v_i[1].items())), shortest_paths)
        shortest_paths = map(lambda v_i: (v_i[0], map(lambda neighbours: (neighbours[0], min(neighbours[1], self.N_classes - 1)), v_i[1])), shortest_paths)
        for source, neighbors in shortest_paths:
            neighbors = filter(lambda neighbor: (not (neighbor[0] == source)) and (neighbor[1] > 0), neighbors)
            # print(len(zip(*neighbors)))
            neighbors = list(zip(*neighbors))
            if len(neighbors) == 0:
                indices, distances = [], []
            else:               
                indices, distances = neighbors
        
            self.k_hop_neighbors_target_indices[source] = torch.tensor(list(indices), dtype=torch.long)
            self.k_hop_neighbors_distances[source] = torch.tensor(list(distances), dtype=torch.long) - 1 # Map distance 1 to class 0
            self.k_hop_neighbors_source_indices[source] = torch.ones(len(indices), dtype=torch.long) * source

    def make_loss(self, embeddings : Tensor):
        source_node_indices = random.sample(self.indices, k=self.N_sampled_nodes)

        source_nodes = torch.cat([self.k_hop_neighbors_source_indices[i] for i in source_node_indices])
        neighbor_nodes = torch.cat([self.k_hop_neighbors_target_indices[i] for i in source_node_indices])
        distances = torch.cat([self.k_hop_neighbors_distances[i] for i in source_node_indices])
        # assert not torch.isnan(embeddings).any(), "embeddings contains NaN values"
        # assert not torch.isinf(embeddings).any(), "embeddings contains infinite values"
        # assert not torch.isnan(source_nodes).any(), "source_nodes contains NaN values"
        # assert not torch.isnan(neighbor_nodes).any(), "neighbor_nodes contains NaN values"
        # assert not torch.isnan(distances).any(), "distances contains NaN values"

        vi_embeddings = embeddings[source_nodes]
        neighbor_embeddings = embeddings[neighbor_nodes]

        predicted_distance = self.decoder((vi_embeddings - neighbor_embeddings).abs())
        # assert not torch.isnan(predicted_distance).any(), "predicted_distance contains NaN values"
        # assert not torch.isinf(predicted_distance).any(), "predicted_distance contains infinite values"

        loss = self.loss(input=predicted_distance, target=distances)
        return loss

@gin.configurable
class PairwiseAttrSim(BasicPretextTask):
    '''
    Proposed by Jin, Wei, et al. "Self-supervised learning on graphs: Deep insights and new direction." arXiv preprint arXiv:2006.10141 (2020).
    '''
    def __init__(self, k_largest : int, **kwargs):
        super().__init__(**kwargs)

        self.k_largest = k_largest
        self.loss = nn.MSELoss(reduction='mean')
        self.decoder = torch.nn.Sequential(
            Linear(self.encoder.out_channels, 1),
        )
        # Cosine similarity undefined for null vector
        self.null_mask = ~torch.isclose(input=self.data.x.sum(dim=1), other=torch.tensor(0.))
        X = self.data.x[self.null_mask]

        similarities = pairwise_cosine_similarity(X)
        similarities.fill_diagonal_(similarities.min() - 1)
        T_s_ = similarities.topk(k=k_largest, largest=True, dim=1)
        T_d_ = similarities.topk(k=k_largest, largest=False, dim=1)

        self.T_s = T_s_.indices.view(-1)
        self.T_d = T_d_.indices.view(-1)
        self.top_sims = torch.cat([
            T_s_.values.view(-1),
            T_d_.values.view(-1)
        ])       
        self.v_indices = torch.arange(0, self.data.num_nodes).view((-1, 1)).repeat((1, k_largest)).view(-1)
        

    def make_loss(self, embeddings : Tensor):
        X_hat = embeddings[self.null_mask]

        highest_similarities_pred = self.decoder((X_hat[self.v_indices, :] - X_hat[self.T_s, :]).abs())
        smallest_similarities_pred = self.decoder((X_hat[self.v_indices, :] - X_hat[self.T_d, :]).abs())

        predicted = torch.cat([
            highest_similarities_pred,
            smallest_similarities_pred
        ]).squeeze()
        loss = self.loss(input=predicted, target=self.top_sims)
        return loss
    
