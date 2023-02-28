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

# ==================================================== #
# ============= Auxiliary property-based ============= # 
# ==================================================== #
@gin.configurable
class NodeClusteringWithAlignment(BasicPretextTask):
    def __init__(self, n_clusters : int, random_state : int = 3364, **kwargs):
        super().__init__(**kwargs)

        # Step 0: Setup
        train_mask = self.data.train_mask
        X, y = self.data.x, self.data.y
        num_classes = y.unique().shape[0]
        feat_dim = X.shape[1]
        centroids_labeled = torch.zeros((num_classes, feat_dim))

        # Step 1: Compute centroids in each cluster by the mean in each class
        for cn in range(num_classes):
            lf = X[train_mask]
            ll = y[train_mask]
            centroids_labeled[cn] = lf[ll == cn].mean(axis=0)

        # Step 2: Set cluster labels for each node
        cluster_labels = torch.ones(y.shape, dtype=torch.int64) * -1
        cluster_labels[train_mask] = y[train_mask]

        # Step 3: Train KMeans on all points
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)

        # Step 4: Perform alignment mechanism
        # See https://arxiv.org/pdf/1902.11038.pdf and 
        # https://github.com/Junseok0207/M3S_Pytorch/blob/master/models/M3S.py for code implementaiton.
        # 1) Compute its centroids
        # 2) Find cluster closest to the centroid computed in step 1
        # 3) Assign all unlabeled nodes to that closest cluster.
        for cn in range(n_clusters):
            # v_l
            centroids_unlabeled = X[torch.logical_and(torch.tensor(kmeans.labels_ == cn), ~train_mask)].mean(axis=0)

            # Equation 5
            label_for_cluster = np.linalg.norm(centroids_labeled - centroids_unlabeled, axis=1).argmin()
            for node in np.where(kmeans.labels_ == cn)[0]:
                if not train_mask[node]:
                    cluster_labels[node] = label_for_cluster
                
        self.pseudo_labels = cluster_labels
        self.decoder = Linear(self.encoder.out_channels, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()


    def make_loss(self, embeddings : Tensor) -> float:
        y_hat = self.decoder(embeddings)
        return self.loss(input=y_hat[~self.train_mask], target=self.pseudo_labels[~self.train_mask])


@gin.configurable
class GraphPartition(BasicPretextTask):
    def __init__(self, n_parts : int, **kwargs):
        super().__init__(**kwargs)
        sparse_matrix = to_scipy_sparse_matrix(self.data.edge_index)
        node_num = sparse_matrix.shape[0]
        adj_list = [[] for _ in range(node_num)]
        for i, j in zip(sparse_matrix.row, sparse_matrix.col):
            if i == j:
                continue
            adj_list[i].append(j)

        _, ss_labels =  pymetis.part_graph(adjacency=adj_list, nparts=n_parts)

        self.pseudo_labels = torch.tensor(ss_labels, dtype=torch.int64)
        self.decoder = Linear(self.encoder.out_channels, n_parts)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def make_loss(self, embeddings: Tensor) -> float:
        y_hat = self.decoder(embeddings)
        return self.loss(input=y_hat, target=self.pseudo_labels)