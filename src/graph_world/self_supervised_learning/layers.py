import torch
from torch.nn import Bilinear, Linear, Tanh
from torch import Tensor

# Simplified version of a pairwise Neural Tensor Network (NTN)
# The pairwise computations are focused around edges
# - Given an edge (u,j), the embeddings of node u and j will be used in the pairwise computation
# - Usages: See the class generation-based.DenoisingLinkReconstruction
class NeuralTensorLayer(torch.nn.Module):
    '''
    Neural Tensor Layer proposed by the authors of the Neural Tensor Network:
        Socher, Richard, et al. "Reasoning with neural tensor networks for knowledge base completion." Advances in neural information processing systems 26 (2013).
    '''
    def __init__(self, in1_features : int, in2_features : int, out_features : int, activation = Tanh()):
        super().__init__()
        self.bilinear_layer = Bilinear(in1_features, in2_features, out_features, bias=False)
        self.linear_layer = Linear(in1_features + in2_features, out_features, bias=True)   
        self.activation = activation


    def forward(self, x1 : Tensor, x2 : Tensor) -> Tensor:
        z = self.bilinear_layer(x1, x2) + self.linear_layer(torch.cat((x1,x2), dim=1))
        return self.activation(z)
