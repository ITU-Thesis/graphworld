from .__types import *
from abc import ABC, abstractclassmethod
from torch.nn import Module
from torch import Tensor, FloatTensor, DoubleTensor
import torch


class BasicPretextTask(Module, ABC):
    def __init__(self, data : InputGraph, encoder : Module, train_mask : Tensor, epochs : int, **kwargs): # **kwargs is needed
        super().__init__()
        self.data = data.clone()
        self.data_test = self.data.clone()
        self.encoder = encoder
        self.epochs = epochs # How many epochs make_loss can be expected to be called
        self.train_mask = train_mask

    @property
    def input_dim(self):
        return self.data.x.shape[1]

    # Override this function to return the pretext task loss
    # The embeddings for the downstream task is given, to be used
    # when the input graph is the same for downstream/pretext tasks
    @abstractclassmethod
    def make_loss(self, embeddings : Tensor) -> Union[FloatTensor, DoubleTensor]:
        pass

    # Override this method if the embeddings from the downstream task
    # does not take the original raw data as input
    # - For example in some URL based methods, augmentations are expected even for the downstream task
    def get_downstream_embeddings(self) -> Tensor:
        return self.encoder(self.data.x, self.data.edge_index)
    
    # Override this method if the embedding size does not match
    # the output of the graph encoder.
    # - For example in some siamese networks where embeddings are concatted
    def get_embedding_dim(self) -> int:
        return self.encoder.out_channels
    
    # def get_pretext_embeddings(self, downstream_embeddings: Tensor = None):
    #     if downstream_embeddings is not None:
    #         return downstream_embeddings
    #     else:
    #         return self.get_downstream_embeddings()
    

# Used if there is no pretext task or it is set to None
class IdentityPretextTask(BasicPretextTask):
    def make_loss(self, embeddings : Tensor) -> Union[FloatTensor, DoubleTensor]:
        return 0
