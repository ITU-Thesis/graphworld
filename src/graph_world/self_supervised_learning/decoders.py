import torch
from torch.nn import Bilinear, Linear
from torch import Tensor

# Simplified version of a pairwise Neural Tensor Network (NTN)
# The pairwise computations are focused around edges
# - Given an edge (u,j), the embeddings of node u and j will be used in the pairwise computation
# - Usages: See the class generation-based.DenoisingLinkReconstruction
class NTNDecoder(torch.nn.Module):
    def __init__(self, in1 : int, in2 : int, out : int):
        super().__init__()
        self.b = Bilinear(in1, in2, out, bias=False) # bilinear tensor product
        self.v = Linear(in1 + in2, out, bias=True) # linear transformation + bias
        self.activation = torch.nn.Tanh()
    
    def forward(self, z: Tensor, edge_index: Tensor):
        e1 = z[edge_index[0]]
        e2 = z[edge_index[1]]
        z1 = self.b(e1, e2) + self.v(torch.cat((e1,e2), dim=1))
        return self.activation(z1)