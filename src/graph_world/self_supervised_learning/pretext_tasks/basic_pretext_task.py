from .__types import *
from abc import ABC, abstractclassmethod
from torch.nn import Module
from torch import Tensor, FloatTensor, DoubleTensor


class BasicPretextTask(ABC):
    def __init__(self, data : InputGraph, encoder : Module, train_mask : Tensor, epochs : int, **kwargs): # **kwargs is needed
        self.data = data.clone()
        self.encoder = encoder
        self.epochs = epochs # How many epochs make_loss can be expected to be called
        self.train_mask = train_mask
        self.decoder = Module()

    # Override this function to return the pretext task loss
    # The embeddings for the downstream task is given, to be used
    # when the input graph is the same for downstream/pretext tasks
    @abstractclassmethod
    def make_loss(self, embeddings : Tensor) -> Union[FloatTensor, DoubleTensor]:
        pass

