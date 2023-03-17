
import torch
from torch.nn import PReLU
from torch import nn
from torch_geometric.utils import k_hop_subgraph
from torchmetrics.functional import pairwise_cosine_similarity

from .basic_pretext_task import BasicPretextTask
from ..graph import SubGraph, SubGraphs
from torch_geometric.nn import global_mean_pool

from .auxiliary_property_based import CentralityScore, GraphPartition
from .generation_based import DenoisingLinkReconstruction

# class GMI(BasicPretextTask):
#     '''
#     Introduced in Peng, Zhen, et al. "Graph representation learning via graphical mutual information maximization." Proceedings of The Web Conference 2020. 2020.
#     See https://github.com/zpeng27/GMI for their original code.

#     Version
#     -------
#     Implements the adaptive (GMI-adaptive). Could also implement the mean (GMI-mean) by setting w_ij=1/i_n as in the paper.
#     '''
#     def __init__(self, k:int, **kwargs):
#         super().__init__(**kwargs)
#         self.prelu = PReLU()

#         self.decoder = nn.Sequential(
#             nn.Bilinear(
#                 in1_features=self.input_dim, 
#                 in2_features=self.get_embedding_dim, 
#                 out_features=1, 
#                 bias=False
#             ),
#             nn.Sigmoid()
#         )
        
#         # Extract neighbor indices and edges for each node's k-hop neighborhood
#         k_hop_subgraphs = [
#              k_hop_subgraph(i, num_hops=k, edge_index=self.data.edge_index, relabel_nodes=True, flow='source_to_target')[0:2]
#              for i in range(self.data.num_nodes)
#         ]
#         subgraphs_data = [
#              SubGraph(node_indices=subgraph_nodes_and_edges[0], data=self.data, subgraph_edges=subgraph_nodes_and_edges[1])
#              for subgraph_nodes_and_edges in k_hop_subgraphs
#         ]
#         self.subgraphs = SubGraphs(subgraphs=subgraphs_data)


#     def __compute_weights(self, embeddings: torch.tesnor):
#         '''
#         Adaptive version of GMI.
#         '''
#         subgraphs = self.subgraphs.subgraph_batches
        
#         h_neighbor = self.encoder(data=subgraphs.x, edge_index=subgraphs.edge_index)
#         neighbor_summaries = self.prelu(global_mean_pool(subgraphs.x, batch=subgraphs.batch))

#         return torch.sigmoid(pairwise_cosine_similarity(embeddings))



#     def make_loss(self, embeddings):

#         # adj_ori: Dense adjacency matrix
#         # sp_adj: Sparse version of the adjacency matrix

#         # forward(seq1, adj_ori, neg_num, adj, samp_bias1, samp_bias2):
#         # model(seq1=features, adj_ori=adj_ori, neg_num=args.negative_num, adj==sp_adj, samp_bias1=None, samp_bias2=None) 
        


#         # A = to_dense_adj(self.data.edge_index)
#         # h_neighbor = self.prelu(torch.matmul(A, embeddings))




#         ...



class HuEtAL(BasicPretextTask):
    def __init__(self, n_parts : int, edge_mask_ratio : float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.pretext_tasks = nn.ModuleDict({
            'denoising_link_reconstruction': DenoisingLinkReconstruction(edge_mask_ratio=edge_mask_ratio, **kwargs),
            'centrality_score_ranking': CentralityScore(**kwargs),
            'cluster_preserving': GraphPartition(n_parts=n_parts, **kwargs)
        })

    
    def make_loss(self, embeddings, **kwargs):
        return 1/3 * sum(map(lambda task: task.make_loss(embeddings, **kwargs), self.pretext_tasks))