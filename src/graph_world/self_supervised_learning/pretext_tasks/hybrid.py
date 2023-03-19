
from typing import List, Set, Union
import torch
from torch import nn
# from torchmetrics.functional import pairwise_cosine_similarity


from ..augmentation import node_feature_shuffle
from .basic_pretext_task import BasicPretextTask
from torch_geometric.nn import global_mean_pool

from .auxiliary_property_based import CentralityScoreRanking, GraphPartitioning
from .generation_based import DenoisingLinkReconstruction
import random
from torch_geometric.data import Data
import gin
from ..loss import jensen_shannon_loss
from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph, get_laplacian
from .utils import get_exact_ppr_matrix, pairwise_cosine_similarity

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


@gin.configurable
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

@gin.configurable
class G_Zoom(BasicPretextTask):
    '''
    Proposed in:
        Zheng, Yizhen, et al. "Toward Graph Self-Supervised Learning With Contrastive Adjusted Zooming." IEEE Transactions on Neural Networks and Learning Systems (2022).
    '''
    
    class Decoder(nn.Module):
        def __init__(self, in1 : int, in2 : int, **kwargs):
            super().__init__(**kwargs)
            self.bilinear = torch.nn.Bilinear(in1, in2, 1)
            

        def forward(self, input1 : Tensor, input2 : Tensor):
            x = self.bilinear(input1, input2)
            return torch.sigmoid(x)

    def __init__(self, B_perc : float, k : int, P_perc : float, alpha : float, alpha_loss : float, beta_loss : float, gamma_loss : float, **kwargs):
        '''
        args
        ----
        micro_meso_macro_weights:
            Weights of the micro, mseo, and macro level contrastive learnings.
        '''

        super().__init__(**kwargs)
        assert P_perc >= 1.
        self.alpha_loss = alpha_loss
        self.beta_loss = beta_loss
        self.gamma_loss = gamma_loss
        
        self.decoder = G_Zoom.Decoder(self.get_embedding_dim(), self.get_embedding_dim())
        self.B = int(self.data.num_nodes * B_perc)
        self.k = k
        self.P = max(self.data.num_nodes, int(P_perc * self.B * k))
        self.batch_indices = torch.arange(0, self.B).view(-1, 1).repeat((1, k)).view(-1)

        # Compute PPR
        PPR_matrix = get_exact_ppr_matrix(data=self.data, alpha=alpha)
        G_tilde_edges, G_tilde_weights = dense_to_sparse(PPR_matrix)
        
        self.G = self.data
        self.G_tilde = Data(x=self.data.x, edge_index=G_tilde_edges, edge_weight=G_tilde_weights)

        # Compute neighborhood register
        importance_matrix = PPR_matrix
        importance_matrix.fill_diagonal_(importance_matrix.min() - 1)
        self.R = self.__get_neighborhood_register(I=importance_matrix, k=k)
        self.same_batch = torch.ones(self.B)
        torch.autograd.set_detect_anomaly(True)
        
    def __get_neighborhood_register(self, I : Tensor, k : int) -> Tensor:
        '''
        Compute the neighborhood register given the importance matrix I and the number of k important neighbors for each node.

        Returns
        -------

        The neighborhood register
        '''
        R = I.topk(k=k, dim=1).indices

        return R
        
    def __graph_samplig(self) -> Union[List[int], List[int]]:
        '''
        Sample the input graph G giving the first augmented graph G1.
        
        Steps
        -----
        1: Sample a batch of B random nodes in the graph.
        2: For each target node sample its top k most important neighbor nodes from the neighborhood register.
        3: Sample P - k random nodes in the graph which was not sampled in step 1 and 2.
        4: Return the target nodes and all nodes from step 1, 2, and 3.
        '''
        all_nodes = {*range(self.data.num_nodes)}

        # Step 1
        target_nodes = random.sample(all_nodes, k=self.B)

        # Step 2
        top_k_neighbors = set(self.R[target_nodes].flatten().detach().tolist())
        assert top_k_neighbors.issubset(all_nodes), 'not subset' # TODO: Remove when we have tested on a couple of graphs

        # Step 3
        nodes_not_a_neighbor = all_nodes - top_k_neighbors
        random_selected_nodes = set(random.sample(nodes_not_a_neighbor, k=self.P-(self.B * self.k)))

        subgraph_nodes = top_k_neighbors | random_selected_nodes
        
        # Step 4
        return list(target_nodes), list(subgraph_nodes)

    def micro_contrastiveness_loss(self, H1 : Tensor, H2 : Tensor, target_nodes : Set[int]) -> Tensor:
        H1_t, H2_t = H1[target_nodes], H2[target_nodes]

        adjust_constant = torch.tensor(1).exp() # Adjust for cosine_sim(vi, vi) in same view

        # Exponential of all similarities
        similarities_exp = {
            '(H1, H1)' : pairwise_cosine_similarity(H1_t, H1_t).exp(),
            '(H1, H2)' : pairwise_cosine_similarity(H1_t, H2_t).exp(),
            '(H2, H2)' : pairwise_cosine_similarity(H2_t, H2_t).exp()
        }

        similarity_exp_sums = {
            '(H1, H1)' : similarities_exp['(H1, H1)'].sum(dim=1) - adjust_constant,
            '(H1, H2)' : similarities_exp['(H1, H2)'].sum(dim=1),                   # Include cosine_sim(vi, vi) in different views
            '(H2, H2)' : similarities_exp['(H2, H2)'].sum(dim=1) - adjust_constant
        }


        positive_pairs = similarities_exp['(H1, H2)'].diag()

        H1_H2 = torch.log(positive_pairs / (similarity_exp_sums['(H1, H2)'] + similarity_exp_sums['(H1, H1)'] + 1e-8))
        H2_H1 = torch.log(positive_pairs / (similarity_exp_sums['(H1, H2)'] + similarity_exp_sums['(H2, H2)'] + 1e-8))
        
        L_micro = -(1/(2 * self.B)) * (H1_H2.sum() + H2_H1.sum())
        return L_micro

    def meso_contrastiveness_loss(self, H1 : Tensor, H2 : Tensor, H_tilde : Tensor, target_nodes : Set[int]) -> Tensor:
        H1_t, H2_t, H_tilde_t = H1[target_nodes], H2[target_nodes], H_tilde[target_nodes]
        top_k_neighbors = self.R[target_nodes].view(-1)                          # Map target nodes to its top-k nodes


        n1 = global_mean_pool(x=H1[top_k_neighbors], batch=self.batch_indices)
        n2 = global_mean_pool(x=H2[top_k_neighbors], batch=self.batch_indices)

        H1_N2 = jensen_shannon_loss(positive_instance=self.decoder(H1_t, n2), negative_instance=self.decoder(H_tilde_t, n2), reduction='sum')
        H2_N1 = jensen_shannon_loss(positive_instance=self.decoder(H2_t, n1), negative_instance=self.decoder(H_tilde_t, n1), reduction='sum')
        L_meso = (1/(2 * self.B)) * (H1_N2 + H2_N1) # jensen_shannon already negative
        return L_meso

    def macro_contrastiveness_loss(self, H1 : Tensor, H2 : Tensor, H_tilde : Tensor, target_nodes : Set[int]) -> torch.Tensor:
        H1_t, H2_t, H_tilde_t = H1[target_nodes], H2[target_nodes], H_tilde[target_nodes]
        s1, s2 = H1.mean(dim=0), H2.mean(dim=0)
        s1, s2 = s1.repeat((self.B, 1)), s2.repeat((self.B, 1))
        
        H1_S2 = jensen_shannon_loss(positive_instance=self.decoder(H1_t, s2), negative_instance=self.decoder(H_tilde_t, s2), reduction='sum')
        H2_S1 = jensen_shannon_loss(positive_instance=self.decoder(H2_t, s1), negative_instance=self.decoder(H_tilde_t, s1), reduction='sum')
        L_macro = (1/(2*self.B)) * (H1_S2 + H2_S1)
        return L_macro
        
    def make_loss(self, embeddings, **kwargs):
        target_nodes, subgraph_nodes = self.__graph_samplig()

        G1_edge_index, *_ = subgraph(subset=subgraph_nodes, edge_index=self.G.edge_index, relabel_nodes=False)
        G2_edge_index, G2_weights = subgraph(subset=subgraph_nodes, edge_index=self.G_tilde.edge_index, edge_attr=self.G_tilde.edge_weight, relabel_nodes=False)

        H1 = self.encoder(x=self.data.x, edge_index=G1_edge_index)
        H2 = self.encoder(x=self.data.x, edge_index=G2_edge_index, edge_weight=G2_weights)

        X_tilde, edge_index_tilde = node_feature_shuffle(node_features=self.data.x, edge_index=G1_edge_index)
        H_tilde = self.encoder(x=X_tilde, edge_index=edge_index_tilde)

        L_micro = self.micro_contrastiveness_loss(H1=H1, H2=H2, target_nodes=target_nodes)
        L_meso = self.meso_contrastiveness_loss(H1=H1, H2=H2, H_tilde=H_tilde, target_nodes=target_nodes)
        L_macro = self.macro_contrastiveness_loss(H1=H1, H2=H2, H_tilde=H_tilde, target_nodes=target_nodes)
        L = self.alpha_loss * L_micro + self.beta_loss * L_meso + self.gamma_loss * L_macro
        return L
    
    def get_downstream_embeddings(self) -> Tensor:
        H1 = self.encoder(x=self.G.x, edge_index=self.G.edge_index)
        H2 = self.encoder(x=self.G_tilde.x, edge_index=self.G_tilde.edge_index, edge_weight=self.G_tilde.edge_weight)
        return H1 + H2

        