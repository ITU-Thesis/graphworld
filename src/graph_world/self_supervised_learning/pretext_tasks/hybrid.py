
from typing import List
import torch
from torch.nn import PReLU
from torch import nn
from torch_geometric.utils import k_hop_subgraph
from torchmetrics.functional import pairwise_cosine_similarity

from .basic_pretext_task import BasicPretextTask
from ..graph import SubGraph, SubGraphs
from torch_geometric.nn import global_mean_pool

from .auxiliary_property_based import CentralityScoreRanking, GraphPartitioning
from .generation_based import DenoisingLinkReconstruction
import random

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
            'centrality_score_ranking': CentralityScoreRanking(**kwargs),
            'cluster_preserving': GraphPartitioning(n_parts=n_parts, **kwargs)
        })

    
    def make_loss(self, embeddings, **kwargs):
        return 1/3 * sum(map(lambda task: task.make_loss(embeddings, **kwargs), self.pretext_tasks))
    

# Augmentations: Graph diffusion + sampling (2 augmented views)
# G1 sample from original graph | G2 sampled from diffused graph
# H1                            | H2
# n_1^t                         | n_2^t                         Pooling layer from top-k neighbors for each node
# s_1                           | s_2                           Subgraph level representaiton. Readout embeddings H1 and H2.

class G_Zoom(BasicPretextTask):
    '''
    Proposed in:
        Zheng, Yizhen, et al. "Toward Graph Self-Supervised Learning With Contrastive Adjusted Zooming." IEEE Transactions on Neural Networks and Learning Systems (2022).
    '''
    def __init__(self, micro_meso_macro_weights : List[float], **kwargs):
        '''
        args
        ----
        micro_meso_macro_weights:
            Weights of the micro, mseo, and macro level contrastive learnings.
        '''
        super().__init__(**kwargs)
        assert len(micro_meso_macro_weights) == 3



    def graph_samplig(self, k : int, P : int):
        '''
        Sample the input graph G giving the first augmented graph G1.
        '''
        
        all_nodes = [*range(self.data.num_nodes)]

        # Target nodes
        target_nodes = random.sample(all_nodes, k=k)

        # Pick out nodes indices
        target_neighbor_nodes = set([
            k_hop_subgraph(i, num_hops=k, edge_index=self.data.edge_index, relabel_nodes=True, flow='source_to_target')[0]
             for i in target_nodes
        ])







        
        
        

        

    def make_loss(self, embeddings, **kwargs):
        ...
