from torch.nn import Linear
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn.functional as F
import gin
from abc import ABC, abstractclassmethod
from .__types import *
from torch.nn import Module
from torch import Tensor
from sklearn.cluster import KMeans
import pymetis
from torch_geometric.utils.convert import to_scipy_sparse_matrix

class BasicPretextTask(ABC):
    def __init__(self, data : InputGraph, encoder : Module, train_mask : Tensor, **kwargs): # **kwargs is needed
        self.data = data.clone()
        self.encoder = encoder
        self.train_mask = train_mask
        self.decoder = None

    # Override this function to return the pretext task loss
    # The embeddings for the downstream task is given, to be used
    # when the input graph is the same for downstream/pretext tasks
    @abstractclassmethod
    def make_loss(self, embeddings : Tensor) -> float:
        pass



# ============================================ #
# ============= Generation-based ============= # 
# ============================================ #

# ------------- Feature generation ------------- #
@gin.configurable
class AttributeMask(BasicPretextTask):
    def __init__(self, node_mask_ratio : float = 0.1, **kwargs):
        super().__init__(**kwargs)

        # Crea mask of subset of unlabeled nodes
        all = np.arange(self.data.x.shape[0])
        unlabeled = all[~self.train_mask]
        perm = np.random.permutation(unlabeled)
        self.masked_nodes = perm[: int(len(perm)*node_mask_ratio)]

        # Generate pseudo labels and mask input features
        # We employ PCA to pseudo labels/predictions
        # if features are high-dimensional
        self.pseudo_labels = self.data.x.clone()
        self.data.x[self.masked_nodes] = torch.zeros(self.data.x.shape[1])
        if self.pseudo_labels.shape[1] > 256:
            pca = PCA(n_components=256)
            self.pseudo_labels = pca.fit_transform(self.pseudo_labels)
        self.pseudo_labels = torch.FloatTensor(self.pseudo_labels[self.masked_nodes]).float()

        # Specify pretext decoder
        self.decoder = Linear(self.encoder.out_channels, self.pseudo_labels.shape[1])

    # Run masked input through graph encoder instead of using the original embeddings
    def make_loss(self, embeddings : Tensor) -> float:
        z = self.encoder(self.data.x, self.data.edge_index)
        y_hat = (self.decoder(z[self.masked_nodes]))
        loss = F.mse_loss(y_hat, self.pseudo_labels, reduction='mean')
        return loss


@gin.configurable
class CorruptedFeaturesReconstruction(BasicPretextTask):
    def __init__(self, feature_corruption_ratio : float = 0.1, 
                 partial_feature_reconstruction : bool =True, **kwargs):
        super().__init__(**kwargs)

        # Create Mask of subset of feature columns
        f_cols = np.arange(self.data.x.shape[1])
        perm = np.random.permutation(f_cols)
        masked_f_cols = perm[: int(len(perm)*feature_corruption_ratio)]

        # Create pseudo labels
        self.pseudo_labels = self.data.x.clone()
        if partial_feature_reconstruction:
            self.pseudo_labels = self.pseudo_labels[:, masked_f_cols]

         # Mask input features
        self.data.x[:,masked_f_cols] = 0

         # Specify pretext decoder
        self.decoder = Linear(self.encoder.out_channels, self.pseudo_labels.shape[1])

    # Run masked input through graph encoder instead of using the original embeddings
    def make_loss(self, embeddings : Tensor) -> float:
        z = self.encoder(self.data.x, self.data.edge_index)
        y_hat = (self.decoder(z))
        loss = F.mse_loss(y_hat, self.pseudo_labels, reduction='mean')
        return loss
    

@gin.configurable
class CorruptedEmbeddingsReconstruction(BasicPretextTask):
    def __init__(self, embedding_corruption_ratio : float = 0.1, 
                 partial_embedding_reconstruction : bool = True, **kwargs):
        super().__init__(**kwargs)

        self.partial_embedding_reconstruction = partial_embedding_reconstruction

        # Create Mask of subset of embedding columns
        embedding_cols = np.arange(self.encoder.out_channels)
        perm = np.random.permutation(embedding_cols) # Likely not needed
        self.masked_embedding_cols = perm[: int(len(perm)*embedding_corruption_ratio)]
        self.mask = torch.eye(self.encoder.out_channels)
        self.mask[self.masked_embedding_cols, self.masked_embedding_cols] = 0

        # Specify pretext decoder
        out = len(self.masked_embedding_cols) if partial_embedding_reconstruction else self.encoder.out_channels
        self.decoder = Linear(self.encoder.out_channels, out)

    # Mask embeddings and reconstruct with decoder
    def make_loss(self, embeddings : Tensor) -> float:
        masked_embeddings = torch.matmul(embeddings, self.mask)
        y_hat = (self.decoder(masked_embeddings))
        if self.partial_embedding_reconstruction:
            pseudo_labels = embeddings[:, self.masked_embedding_cols]
        else:
            pseudo_labels = embeddings
        return F.mse_loss(y_hat, pseudo_labels, reduction='mean')
    

@gin.configurable
class AutoEncoding(BasicPretextTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = Linear(self.encoder.out_channels, self.data.x.shape[1])
        self.pseudo_labels = self.data.x

    # Directly reconstruct input features from embedding
    def make_loss(self, embeddings : Tensor) -> float:
        y_hat = self.decoder(embeddings)
        return F.mse_loss(y_hat, self.pseudo_labels)





# ------------- Structure generation ------------- #


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