from torch.nn import Linear
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn.functional as F
import gin


class BasicPretextTask:
    def __init__(self, data, encoder, train_mask, **kwargs):
        self.data = data.clone()
        self.encoder = encoder
        self.train_mask = train_mask
        self.decoder = None

    # Override this function to return the pretext task loss
    # The embeddings for the downstream task is given, to be used
    # when the input graph is the same for downstream/pretext tasks
    def make_loss(self, embeddings):
        raise NotImplementedError
    
    # Override this function to return a  list of strings of the hyperparameters
    # used by this task. This is needed because the pretext tasks and GNN encoder
    # uses the same dictionary for hyper params, and the GNN models cannot handle the
    # new parameters.
    @staticmethod
    def used_hparams():
        raise NotImplementedError




# ============================================ #
# ============= Generation-based ============= # 
# ============================================ #

# ------------- Feature generation ------------- #
@gin.configurable
class AttributeMask(BasicPretextTask):
    def __init__(self, mask_ratio=0.1, **kwargs):
        super().__init__(**kwargs)

        # Mask subset of unlabeled nodes
        all = np.arange(self.data.x.shape[0])
        unlabeled = all[~self.train_mask]
        perm = np.random.permutation(unlabeled)
        self.masked_nodes = perm[: int(len(perm)*mask_ratio)]

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
    def make_loss(self, embeddings):
        z = self.encoder(self.data.x, self.data.edge_index)
        y_hat = (self.decoder(z[self.masked_nodes]))
        loss = F.mse_loss(y_hat, self.pseudo_labels, reduction='mean')
        return loss
    
    @staticmethod
    def used_hparams():
        return ["mask_ratio"]


# ------------- Structure generation ------------- #



# ==================================================== #
# ============= Auxiliary property-based ============= # 
# ==================================================== #

