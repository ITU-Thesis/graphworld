from .__types import *
from abc import ABC, abstractclassmethod
from torch.nn import Module
from torch import Tensor


class BasicPretextTask(ABC):
    def __init__(self, data : InputGraph, encoder : Module, train_mask : Tensor, **kwargs): # **kwargs is needed
        self.data = data.clone()
        self.encoder = encoder
        self.train_mask = train_mask
        self.decoder = Module()

    # Override this function to return the pretext task loss
    # The embeddings for the downstream task is given, to be used
    # when the input graph is the same for downstream/pretext tasks
    @abstractclassmethod
    def make_loss(self, embeddings : Tensor) -> float:
        pass

