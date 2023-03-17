
from src.graph_world.self_supervised_learning.pretext_tasks.basic_pretext_task import BasicPretextTask
import torch
from torch.nn import PReLU
from torch import nn
from torch_geometric.utils import to_dense_adj

class GMI(BasicPretextTask):
    '''
    Introduced in Peng, Zhen, et al. "Graph representation learning via graphical mutual information maximization." Proceedings of The Web Conference 2020. 2020.
    See https://github.com/zpeng27/GMI for their original code.

    Version
    -------
    Implements the adaptive (GMI-adaptive). Could also implement the mean (GMI-mean) by setting w_ij=1/i_n as in the paper.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prelu = PReLU()

        self.decoder = nn.Sequential(
            nn.Bilinear(
                in1_features=self.input_dim, 
                in2_features=self.get_embedding_dim, 
                out_features=1, 
                bias=False
            ),
            nn.Sigmoid()
        )

    def __compute_weights(self, X: torch.tesnor):
        ...


    def make_loss(self, embeddings):

        # adj_ori: Dense adjacency matrix
        # sp_adj: Sparse version of the adjacency matrix

        # forward(seq1, adj_ori, neg_num, adj, samp_bias1, samp_bias2):
        # model(seq1=features, adj_ori=adj_ori, neg_num=args.negative_num, adj==sp_adj, samp_bias1=None, samp_bias2=None) 
        
        A = to_dense_adj(self.data.edge_index)
        h_neighbor = self.prelu(torch.matmul(A, embeddings))


        ...

