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
from torch_geometric.utils import negative_sampling
from ..layers import NeuralTensorLayer
import math

# ------------- Feature generation ------------- #
@gin.configurable
class AttributeMask(BasicPretextTask):
    def __init__(self, node_mask_ratio : float = 0.1, **kwargs):
        super().__init__(**kwargs)

        # Create mask of subset of unlabeled nodes
        all = np.arange(self.data.x.shape[0])
        unlabeled = all[~self.train_mask]
        perm = np.random.permutation(unlabeled)
        self.masked_nodes = perm[: math.ceil(len(perm)*node_mask_ratio)]

        # Generate pseudo labels and mask input features
        # We employ PCA to pseudo labels/predictions
        # if features are high-dimensional
        self.pseudo_labels = self.data.x.clone()
        self.augmentation = self.data.x.clone()
        self.augmentation[self.masked_nodes] = torch.zeros(self.augmentation.shape[1])
        if self.pseudo_labels.shape[1] > 256:
            pca = PCA(n_components=256)
            self.pseudo_labels = pca.fit_transform(self.pseudo_labels)
        self.pseudo_labels = torch.FloatTensor(self.pseudo_labels[self.masked_nodes]).float()

        # Specify pretext decoder
        self.decoder = Linear(self.encoder.out_channels, self.pseudo_labels.shape[1])

    # Run masked input through graph encoder instead of using the original embeddings
    def make_loss(self, embeddings : Tensor):
        z = self.encoder(self.augmentation, self.data.edge_index)
        y_hat = (self.decoder(z[self.masked_nodes]))
        loss = F.mse_loss(y_hat, self.pseudo_labels, reduction='mean')
        return loss


@gin.configurable
class CorruptedFeaturesReconstruction(BasicPretextTask):
    def __init__(self, feature_mask_ratio : float = 0.1, 
                 partial_reconstruction : bool =True, **kwargs):
        super().__init__(**kwargs)

        # Create Mask of subset of feature columns
        f_cols = np.arange(self.data.x.shape[1])
        perm = np.random.permutation(f_cols)
        masked_f_cols = perm[: math.ceil(len(perm)*feature_mask_ratio)]

        # Create pseudo labels
        self.pseudo_labels = self.data.x.clone()
        if partial_reconstruction:
            self.pseudo_labels = self.pseudo_labels[:, masked_f_cols]

         # Mask input features
        self.masked_f = self.data.x.clone()
        self.masked_f[:,masked_f_cols] = 0

         # Specify pretext decoder
        self.decoder = Linear(self.encoder.out_channels, self.pseudo_labels.shape[1])

    # Run masked input through graph encoder instead of using the original embeddings
    def make_loss(self, embeddings : Tensor):
        z = self.encoder(self.masked_f, self.data.edge_index)
        y_hat = (self.decoder(z))
        loss = F.mse_loss(y_hat, self.pseudo_labels, reduction='mean')
        return loss
    

@gin.configurable
class CorruptedEmbeddingsReconstruction(BasicPretextTask):
    def __init__(self, embedding_mask_ratio : float = 0.1, 
                 partial_reconstruction : bool = True, **kwargs):
        super().__init__(**kwargs)

        self.partial_reconstruction = partial_reconstruction

        # Create Mask of subset of embedding columns
        embedding_cols = np.arange(self.encoder.out_channels)
        perm = np.random.permutation(embedding_cols) # Likely not needed
        self.masked_embedding_cols = perm[: math.ceil(len(perm)*embedding_mask_ratio)]
        self.mask = torch.eye(self.encoder.out_channels)
        self.mask[self.masked_embedding_cols, self.masked_embedding_cols] = 0

        # Specify pretext decoder
        out = len(self.masked_embedding_cols) if partial_reconstruction else self.encoder.out_channels
        self.decoder = Linear(self.encoder.out_channels, out)

    # Mask embeddings and reconstruct with decoder
    def make_loss(self, embeddings : Tensor):
        masked_embeddings = torch.matmul(embeddings, self.mask)
        y_hat = (self.decoder(masked_embeddings))
        if self.partial_reconstruction:
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
    def make_loss(self, embeddings : Tensor):
        y_hat = self.decoder(embeddings)
        return F.mse_loss(y_hat, self.pseudo_labels)



# ------------- Structure generation ------------- #
@gin.configurable
class GAE(BasicPretextTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pygGAE = pyg_autoencoder.GAE(self.encoder) # Default decoder is InnerProduct
        self.decoder = self.pygGAE.decoder # Needed for optimizer to pull parameters

    # Uses PyG implementation for loss, with negative sampling for non-edges
    def make_loss(self, embeddings : Tensor) -> float:
        return self.pygGAE.recon_loss(embeddings, self.data.edge_index)

@gin.configurable
class VGAE(BasicPretextTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pygVGAE = pyg_autoencoder.VGAE(self.encoder)

        # Transform encoder output to two separate heads for mu and std
        self.muTransform = Linear(self.encoder.out_channels, self.encoder.out_channels)
        self.stdTransform = Linear(self.encoder.out_channels, self.encoder.out_channels)

        # Allows all weights/parameters to be pulled from the decoder variable
        self.decoder = torch.nn.ModuleList(
            [self.muTransform,
            self.stdTransform,
            self.pygVGAE.decoder]
        )
    
    def make_loss(self, embeddings : Tensor):
        # Get variational embedding
        mu, logstd = self.muTransform(embeddings), self.stdTransform(embeddings)
        logstd = logstd.clamp(max=pyg_autoencoder.MAX_LOGSTD)
        variational_embedding = self.pygVGAE.reparametrize(mu, logstd)

        # Compute loss
        recon_loss = self.pygVGAE.recon_loss(variational_embedding, self.data.edge_index)
        kl_loss = self.pygVGAE.kl_loss(mu, logstd)
        return recon_loss + (1/self.data.num_nodes) * kl_loss
    

@gin.configurable
class ARGA(BasicPretextTask):
    def __init__(self, discriminator_lr : float = 0.001, 
                 discriminator_epochs : int = 5, **kwargs):
        super().__init__(**kwargs)

        # Construct discriminator equal to PyG example
        # https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial7/Tutorial7.ipynb#scrollTo=s4mjQFYZviGx
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_channels, 2*self.encoder.out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.encoder.out_channels, 2*self.encoder.out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.encoder.out_channels, 1)
        )

        # Get ARGA implementation from PyG
        self.pygARGA = pyg_autoencoder.ARGA(self.encoder, self.discriminator) # Default decoder is InnerProduct
        self.decoder = self.pygARGA.decoder

        # Discriminator params
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), discriminator_lr)
        self.discriminator_epochs = discriminator_epochs

    def make_loss(self, embeddings : Tensor):
        # In each pretext epoch, we train the discriminator with its own optimizer
        for _ in range(self.discriminator_epochs):
            self.discriminator.train()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self.pygARGA.discriminator_loss(embeddings)
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

        # Then we return the reconstruction loss regularized by the discriminator
        recon_loss = self.pygARGA.recon_loss(embeddings, self.data.edge_index)
        reg_loss = self.pygARGA.reg_loss(embeddings)
        return recon_loss + reg_loss
    

@gin.configurable
class ARGVA(BasicPretextTask):
    def __init__(self, discriminator_lr : float = 0.001, 
                 discriminator_epochs : int = 5, **kwargs):
        super().__init__(**kwargs)

        # Construct discriminator equal to PyG example
        # https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial7/Tutorial7.ipynb#scrollTo=s4mjQFYZviGx
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_channels, 2*self.encoder.out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.encoder.out_channels, 2*self.encoder.out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.encoder.out_channels, 1)
        )

        # Transform encoder output to two separate heads for mu and std
        self.muTransform = Linear(self.encoder.out_channels, self.encoder.out_channels)
        self.stdTransform = Linear(self.encoder.out_channels, self.encoder.out_channels)

        # Get ARGVA implementation from PyG
        self.pygARGVA = pyg_autoencoder.ARGVA(self.encoder, self.discriminator) # Default decoder is InnerProduct

        # Allows all weights/parameters to be pulled from the decoder variable
        self.decoder = torch.nn.ModuleList(
            [self.muTransform,
            self.stdTransform,
            self.pygARGVA.decoder]
        )

        # Discriminator params
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), discriminator_lr)
        self.discriminator_epochs = discriminator_epochs

    def make_loss(self, embeddings : Tensor):
        # Get variational embedding
        mu, logstd = self.muTransform(embeddings), self.stdTransform(embeddings)
        logstd = logstd.clamp(max=pyg_autoencoder.MAX_LOGSTD)
        variational_embedding = self.pygARGVA.reparametrize(mu, logstd)

        # In each pretext epoch, we train the discriminator with its own optimizer
        for _ in range(self.discriminator_epochs):
            self.discriminator.train()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self.pygARGVA.discriminator_loss(variational_embedding)
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

        # Then we return the reconstruction loss regularized by the discriminator and kl divergence
        recon_loss = self.pygARGVA.recon_loss(variational_embedding, self.data.edge_index)
        reg_loss = self.pygARGVA.reg_loss(variational_embedding)
        kl_loss = self.pygARGVA.kl_loss(mu, logstd)
        return recon_loss + (1/self.data.num_nodes) * kl_loss + reg_loss
    

# Include wrapper for superGAT here, since the superGAT model in /models does not use pretext loss
@gin.configurable
class SuperGATSSL(BasicPretextTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.encoder, SuperGAT) # force encoder to be of type superGAT

    # Sum the attention loss for each layer as specified in the superGAT paper
    # Note that the official PyG example does not compute the sum, but just takes the loss from the last layer
    def make_loss(self, embedding: Tensor):
        attention_loss_list = [l.get_attention_loss() for l in self.encoder.convs]
        return sum(attention_loss_list)
    

@gin.configurable
class DenoisingLinkReconstruction(BasicPretextTask):

    def __init__(self, edge_mask_ratio : float = 0.2, **kwargs):
        super().__init__(**kwargs)

        # Create mask to remove edges
        perm = np.random.permutation(np.arange(self.data.edge_index.shape[1]))
        remove_edges = perm[: math.ceil(len(perm)*edge_mask_ratio)]
        edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
        edge_mask[remove_edges] = 0

        # Sample negative edges
        self.neg_edge_index = negative_sampling(self.data.edge_index, num_neg_samples = math.ceil(len(perm)*edge_mask_ratio))

        # Remove of positive edges
        self.removed_edges = self.data.edge_index[:, ~edge_mask]
        self.kept_edges = self.data.edge_index.clone()
        self.kept_edges = self.kept_edges[:, edge_mask]

        # Create NTN decoder (simplified)
        out = self.encoder.out_channels
        self.ntn = NeuralTensorLayer(out, out, 4)
        self.classifier = Linear(4, 1)
        self.decoder = torch.nn.ModuleList([self.ntn, self.classifier])


    # Compute BCE loss of predicting edges for masked edges and negative sampled edges
    # Note that this loss computation is similar to the reconstruction loss of PyG GAE
    # - Difference: Different decoder (NTN) and loss is focused around masks
    def make_loss(self, embedding: Tensor):
        # Run masked input through encoder instead of using the embedding
        z = self.encoder(self.data.x, self.kept_edges)

        # Compute loss for masked edges
        pos_decode_embed = self.ntn(z[self.removed_edges[0]], z[self.removed_edges[1]])
        pos_predict = torch.sigmoid(self.classifier(pos_decode_embed))
        pos_loss = -torch.log(pos_predict).mean()

        # Compute loss for negative sampled edges
        neg_decode_embed = self.ntn(z[self.neg_edge_index[0]], z[self.neg_edge_index[1]])
        neg_predict = torch.sigmoid(self.classifier(neg_decode_embed))
        neg_loss = -torch.log(1 - neg_predict).mean()
        
        return pos_loss + neg_loss
        


# Same as DenoisingLinkReconstruction but different decoder
# This decoder just examines the absolute differences between pairwise embeddings
@gin.configurable
class EdgeMask(DenoisingLinkReconstruction):
    def __init__(self, edge_mask_ratio : float = 0.2, **kwargs):
        super().__init__(edge_mask_ratio, **kwargs)
        self.decoder = Linear(self.encoder.out_channels, 1)

    def make_loss(self, embedding: Tensor):
        # Run masked input through encoder instead of using the embedding
        z = self.encoder(self.data.x, self.data.edge_index)

        # Compute loss for masked edges
        pos_predict = torch.sigmoid(self.decoder(torch.abs(z[self.removed_edges[0] - self.removed_edges[1]])))
        pos_loss = -torch.log(pos_predict).mean()

        # Compute loss for negative sampled edges
        neg_predict = torch.sigmoid(self.decoder(torch.abs(z[self.neg_edge_index[0] - self.neg_edge_index[1]])))
        neg_loss = -torch.log(1 - neg_predict).mean()
        
        return pos_loss + neg_loss