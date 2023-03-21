
from typing import List, Set, Union
import torch
from torch import nn
# from torchmetrics.functional import pairwise_cosine_similarity


from ..augmentation import node_feature_shuffle
from .basic_pretext_task import BasicPretextTask
from torch_geometric.nn import global_mean_pool

from .generation_based import AutoEncoding, CorruptedFeaturesReconstruction, CorruptedEmbeddingsReconstruction
from .auxiliary_property_based import CentralityScoreRanking, GraphPartitioning
from .generation_based import DenoisingLinkReconstruction
import random
from torch_geometric.data import Data
import gin
from ..loss import jensen_shannon_loss
from torch import Tensor
from torch_geometric.utils import dense_to_sparse, subgraph
import torch.nn.functional as F
from .utils import get_exact_ppr_matrix, pairwise_cosine_similarity
from torch_geometric.nn import knn_graph
import copy
from torch_geometric.utils import negative_sampling

@gin.configurable
class HuEtAL(BasicPretextTask):
    def __init__(self, n_parts: int, edge_mask_ratio: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.pretext_tasks = nn.ModuleDict({
            'denoising_link_reconstruction': DenoisingLinkReconstruction(edge_mask_ratio=edge_mask_ratio, **kwargs),
            'centrality_score_ranking': CentralityScoreRanking(**kwargs),
            'cluster_preserving': GraphPartitioning(n_parts=n_parts, **kwargs)
        })

    def make_loss(self, embeddings, **kwargs):
        return sum(map(lambda task: task.make_loss(embeddings, **kwargs), self.pretext_tasks))


@gin.configurable
class MEtAl(BasicPretextTask):
    def __init__(self, partial_reconstruction: bool = False,
                 feature_mask_ratio: float = 0.1,
                 embedding_mask_ratio: float = 0.1,
                 ae_loss_weight: int = 1,
                 fr_loss_weight: int = 1,
                 er_loss_weight: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.weights = [ae_loss_weight, fr_loss_weight, er_loss_weight]
        self.pretext_tasks = nn.ModuleDict({
            'autoencoding': AutoEncoding(**kwargs),
            'corruptedFeaturesReconstruction': CorruptedFeaturesReconstruction(feature_mask_ratio, partial_reconstruction, **kwargs),
            'corruptedEmbeddingsReconstruction': CorruptedEmbeddingsReconstruction(embedding_mask_ratio, partial_reconstruction, **kwargs)
        })

    # We divide by the pretext weight, as the ae/fr/er weights are to be used instead
    # Hence this nullifies the weight multiplication done by the benchmarker during joint training
    def make_loss(self, embeddings):
        loss = 0
        for i, t in enumerate(self.pretext_tasks):
            loss += self.weights[i] * \
                self.pretext_tasks[t].make_loss(embeddings)
        return loss / self.pretext_weight

@gin.configurable
class G_Zoom(BasicPretextTask):
    '''
    Proposed in:
        Zheng, Yizhen, et al. "Toward Graph Self-Supervised Learning With Contrastive Adjusted Zooming." IEEE Transactions on Neural Networks and Learning Systems (2022).
    '''

    class Decoder(nn.Module):
        def __init__(self, in1: int, in2: int, **kwargs):
            super().__init__(**kwargs)
            self.bilinear = torch.nn.Bilinear(in1, in2, 1)

        def forward(self, input1: Tensor, input2: Tensor):
            x = self.bilinear(input1, input2)
            return torch.sigmoid(x)

    def __init__(self, B_perc: float, k: int, P_perc: float, alpha: float, alpha_beta_gamma_weights : List[float], **kwargs):
        super().__init__(**kwargs)
        assert P_perc >= 1.
        self.alpha_loss, self.beta_loss, self.gamma_loss = alpha_beta_gamma_weights

        self.decoder = G_Zoom.Decoder(
            self.get_embedding_dim(), self.get_embedding_dim())
        self.B = int(self.data.num_nodes * B_perc)
        self.k = k
        self.P = max(self.data.num_nodes, int(P_perc * self.B * k))
        self.batch_indices = torch.arange(
            0, self.B).view(-1, 1).repeat((1, k)).view(-1)

        # Compute PPR
        PPR_matrix = get_exact_ppr_matrix(data=self.data, alpha=alpha)
        G_tilde_edges, G_tilde_weights = dense_to_sparse(PPR_matrix)

        self.G = self.data
        self.G_tilde = Data(
            x=self.data.x, edge_index=G_tilde_edges, edge_weight=G_tilde_weights)

        # Compute neighborhood register
        importance_matrix = PPR_matrix
        importance_matrix.fill_diagonal_(importance_matrix.min() - 1)
        self.R = self.__get_neighborhood_register(I=importance_matrix, k=k)
        self.same_batch = torch.ones(self.B)

    def __get_neighborhood_register(self, I: Tensor, k: int) -> Tensor:
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
        # TODO: Remove when we have tested on a couple of graphs
        assert top_k_neighbors.issubset(all_nodes), 'not subset'

        # Step 3
        nodes_not_a_neighbor = all_nodes - top_k_neighbors
        random_selected_nodes = set(random.sample(
            nodes_not_a_neighbor, k=self.P-(self.B * self.k)))

        subgraph_nodes = top_k_neighbors | random_selected_nodes

        # Step 4
        return list(target_nodes), list(subgraph_nodes)

    def micro_contrastiveness_loss(self, H1: Tensor, H2: Tensor, target_nodes: Set[int]) -> Tensor:
        H1_t, H2_t = H1[target_nodes], H2[target_nodes]

        # Adjust for cosine_sim(vi, vi) in same view
        adjust_constant = torch.tensor(1).exp()

        # Exponential of all similarities
        similarities_exp = {
            '(H1, H1)': pairwise_cosine_similarity(H1_t, H1_t).exp(),
            '(H1, H2)': pairwise_cosine_similarity(H1_t, H2_t).exp(),
            '(H2, H2)': pairwise_cosine_similarity(H2_t, H2_t).exp()
        }

        similarity_exp_sums = {
            '(H1, H1)': similarities_exp['(H1, H1)'].sum(dim=1) - adjust_constant,
            # Include cosine_sim(vi, vi) in different views
            '(H1, H2)': similarities_exp['(H1, H2)'].sum(dim=1),
            '(H2, H2)': similarities_exp['(H2, H2)'].sum(dim=1) - adjust_constant
        }

        positive_pairs = similarities_exp['(H1, H2)'].diag()

        H1_H2 = torch.log(
            positive_pairs / (similarity_exp_sums['(H1, H2)'] + similarity_exp_sums['(H1, H1)'] + 1e-8))
        H2_H1 = torch.log(
            positive_pairs / (similarity_exp_sums['(H1, H2)'] + similarity_exp_sums['(H2, H2)'] + 1e-8))

        L_micro = -(1/(2 * self.B)) * (H1_H2.sum() + H2_H1.sum())
        return L_micro

    def meso_contrastiveness_loss(self, H1: Tensor, H2: Tensor, H_tilde: Tensor, target_nodes: Set[int]) -> Tensor:
        H1_t, H2_t, H_tilde_t = H1[target_nodes], H2[target_nodes], H_tilde[target_nodes]
        # Map target nodes to its top-k nodes
        top_k_neighbors = self.R[target_nodes].view(-1)

        n1 = global_mean_pool(x=H1[top_k_neighbors], batch=self.batch_indices)
        n2 = global_mean_pool(x=H2[top_k_neighbors], batch=self.batch_indices)

        H1_N2 = jensen_shannon_loss(positive_instance=self.decoder(
            H1_t, n2), negative_instance=self.decoder(H_tilde_t, n2), reduction='sum')
        H2_N1 = jensen_shannon_loss(positive_instance=self.decoder(
            H2_t, n1), negative_instance=self.decoder(H_tilde_t, n1), reduction='sum')
        # jensen_shannon already negative
        L_meso = (1/(2 * self.B)) * (H1_N2 + H2_N1)
        return L_meso

    def macro_contrastiveness_loss(self, H1: Tensor, H2: Tensor, H_tilde: Tensor, target_nodes: Set[int]) -> torch.Tensor:
        H1_t, H2_t, H_tilde_t = H1[target_nodes], H2[target_nodes], H_tilde[target_nodes]
        s1, s2 = H1.mean(dim=0), H2.mean(dim=0)
        s1, s2 = s1.repeat((self.B, 1)), s2.repeat((self.B, 1))

        H1_S2 = jensen_shannon_loss(positive_instance=self.decoder(
            H1_t, s2), negative_instance=self.decoder(H_tilde_t, s2), reduction='sum')
        H2_S1 = jensen_shannon_loss(positive_instance=self.decoder(
            H2_t, s1), negative_instance=self.decoder(H_tilde_t, s1), reduction='sum')
        L_macro = (1/(2*self.B)) * (H1_S2 + H2_S1)
        return L_macro

    def make_loss(self, embeddings, **kwargs):
        target_nodes, subgraph_nodes = self.__graph_samplig()

        G1_edge_index, *_ = subgraph(subset=subgraph_nodes,
                                     edge_index=self.G.edge_index, relabel_nodes=False)
        G2_edge_index, G2_weights = subgraph(
            subset=subgraph_nodes, edge_index=self.G_tilde.edge_index, edge_attr=self.G_tilde.edge_weight, relabel_nodes=False)

        H1 = self.encoder(x=self.data.x, edge_index=G1_edge_index)
        H2 = self.encoder(
            x=self.data.x, edge_index=G2_edge_index, edge_weight=G2_weights)

        X_tilde, edge_index_tilde = node_feature_shuffle(
            node_features=self.data.x, edge_index=G1_edge_index)
        H_tilde = self.encoder(x=X_tilde, edge_index=edge_index_tilde)

        L_micro = self.micro_contrastiveness_loss(
            H1=H1, H2=H2, target_nodes=target_nodes)
        L_meso = self.meso_contrastiveness_loss(
            H1=H1, H2=H2, H_tilde=H_tilde, target_nodes=target_nodes)
        L_macro = self.macro_contrastiveness_loss(
            H1=H1, H2=H2, H_tilde=H_tilde, target_nodes=target_nodes)
        L = self.alpha_loss * L_micro + self.beta_loss * \
            L_meso + self.gamma_loss * L_macro
        return L

    def get_downstream_embeddings(self) -> Tensor:
        H1 = self.encoder(x=self.G.x, edge_index=self.G.edge_index)
        H2 = self.encoder(x=self.G_tilde.x, edge_index=self.G_tilde.edge_index,
                          edge_weight=self.G_tilde.edge_weight)
        return H1 + H2

@gin.configurable
class MVMI_FT(BasicPretextTask):
    '''
    Proposed in:
        Fan, Xiaolong, et al. "Maximizing mutual information across feature and topology views for learning graph representations." arXiv preprint arXiv:2105.06715 (2021).
    Implementation modified from the authors GitHub: https://github.com/xiaolongo/MaxMIAcrossFT.
    '''

    def summary(self, z) -> Tensor:
        return torch.sigmoid(z.mean(dim=0))

    def corruption(self, x) -> Tensor:
        return x[torch.randperm(x.shape[0])]
    
    def uniform(self, size, tensor):
        if tensor is not None:
            bound = 1.0 / torch.sqrt(size)
            tensor.data.uniform_(-bound, bound)

    def __init__(self, k: int, disagreement_regularization: float, common_representation_regularization: float, **kwargs):
        super().__init__(**kwargs)
        self.A_f = knn_graph(x=self.data.x, k=k)
        self.disagreement_regularization = disagreement_regularization
        self.common_representation_regularization = common_representation_regularization
        
        hidden_dim = self.get_embedding_dim()

        self.encoder_f = copy.deepcopy(self.encoder)    # Feature encoder
        self.encoder_t = copy.deepcopy(self.encoder)    # Topological encoder
        self.encoder_c = self.encoder                   # Common encoder

        self.weight_z_t = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_z_f = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_z_cf = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_z_ct = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        self.mlp_ft = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.PReLU(hidden_dim),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.PReLU(hidden_dim))
        self.mlp_c = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim))
        self.mlp_s = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim))

    def __compute_embeddings(self):
        X, A, A_f = self.data.x, self.data.edge_index, self.A_f

        # Feature view
        pos_z_f = self.encoder_f(X, A_f)
        neg_z_f = self.encoder_f(self.corruption(X), A_f)
        s_f = self.summary(pos_z_f)
        s_f = self.mlp_s(s_f.unsqueeze(0)).squeeze()
        pos_z_f = self.mlp_ft(pos_z_f)
        neg_z_f = self.mlp_ft(neg_z_f)

        # Topology view
        pos_z_t = self.encoder_t(X, A)
        neg_z_t = self.encoder_t(self.corruption(X), A)
        s_t = self.summary(pos_z_t)
        s_t = self.mlp_s(s_t.unsqueeze(0)).squeeze()
        pos_z_t = self.mlp_ft(pos_z_t)
        neg_z_t = self.mlp_ft(neg_z_t)

        # common view
        pos_z_cf = self.encoder_c(X, A_f)
        pos_z_ct = self.encoder_c(X, A)
        pos_z_cft = torch.cat([pos_z_cf, pos_z_ct], dim=-1)
        pos_z_cft = self.mlp_c(pos_z_cft)

        neg_z_cf = self.encoder_c(self.corruption(X), A_f)
        neg_z_ct = self.encoder_c(self.corruption(X), A)
        neg_z_cft = torch.cat([neg_z_cf, neg_z_ct], dim=-1)
        neg_z_cft = self.mlp_c(neg_z_cft)

        s_cft = self.summary(pos_z_cft)

        return {
            's_f': s_f,
            'pos_z_f': pos_z_f,
            'neg_z_f': neg_z_f,
            'pos_z_t': pos_z_t,
            'neg_z_t': neg_z_t,
            's_t': s_t,
            'pos_z_cf': pos_z_cf,
            'pos_z_ct': pos_z_ct,
            'pos_z_cft': pos_z_cft,
            'neg_z_cf': neg_z_cf,
            'neg_z_ct': neg_z_ct,
            'neg_z_cft': neg_z_cft,
            's_cft': s_cft
        }

    def reset_parameters(self):
        self.uniform(self.hidden_dim, self.weight_z_t)
        self.uniform(self.hidden_dim, self.weight_z_f)
        self.uniform(self.hidden_dim, self.weight_z_cf)
        self.uniform(self.hidden_dim, self.weight_z_ct)

    def discriminator_t(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_z_t, s))
        return torch.sigmoid(value)

    def discriminator_f(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_z_f, s))
        return torch.sigmoid(value)

    def discriminator_cf(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_z_cf, s))
        return torch.sigmoid(value)

    def recont_loss(self, z, edge_index):
        pos_edge_index = edge_index
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        pos_reconstructed = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
        neg_reconstructed = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

        loss = jensen_shannon_loss(
            positive_instance=pos_reconstructed,
            negative_instance=neg_reconstructed,
            reduction='mean'
        )
        return loss
    
    def make_loss(self, embeddings, **kwargs):
        E = self.__compute_embeddings()
        assert not torch.isnan(E['s_f']).any(), "s_f contains NaN values"
        assert not torch.isnan(E['pos_z_f']).any(), "pos_z_f contains NaN values"
        assert not torch.isnan(E['neg_z_f']).any(), "neg_z_f contains NaN values"
        assert not torch.isnan(E['pos_z_t']).any(), "pos_z_t contains NaN values"
        assert not torch.isnan(E['neg_z_t']).any(), "neg_z_t contains NaN values"
        assert not torch.isnan(E['s_t']).any(), "s_t contains NaN values"
        assert not torch.isnan(E['pos_z_cf']).any(), "pos_z_cf contains NaN values"
        assert not torch.isnan(E['pos_z_ct']).any(), "pos_z_ct contains NaN values"
        assert not torch.isnan(E['pos_z_cft']).any(), "pos_z_cft contains NaN values"
        assert not torch.isnan(E['neg_z_cf']).any(), "neg_z_cf contains NaN values"
        assert not torch.isnan(E['neg_z_ct']).any(), "neg_z_ct contains NaN values"
        assert not torch.isnan(E['neg_z_cft']).any(), "neg_z_cft contains NaN values"
        assert not torch.isnan(E['s_cft']).any(), "s_cft contains NaN values"

        # feature view
        pos_loss_f = torch.log(
            self.discriminator_f(E['pos_z_f'], E['s_t']) + 1e-7).mean()
        neg_loss_f = torch.log(1 -
                               self.discriminator_f(E['neg_z_f'], E['s_t']) +
                               1e-7).mean()
        mi_loss_f = pos_loss_f + neg_loss_f

        # topology view
        pos_loss_t = torch.log(
            self.discriminator_t(E['pos_z_t'], E['s_f']) + 1e-7).mean()
        neg_loss_t = torch.log((1 - self.discriminator_t(E['neg_z_t'], E['s_f'])) + 1e-7).mean()
        mi_loss_t = pos_loss_t + neg_loss_t

        # common view
        pos_loss_cf = torch.log(self.discriminator_cf(E['pos_z_cft'], E['s_cft']) + 1e-7).mean()
        neg_loss_cf = torch.log((1 - self.discriminator_cf(E['neg_z_cft'], E['s_cft'])) +1e-7).mean()
        mi_loss_cf = pos_loss_cf + neg_loss_cf

        # recont loss
        recont_loss_cftf = self.recont_loss(E['pos_z_cft'], self.A_f)
        recont_loss_cftt = self.recont_loss(E['pos_z_cft'], self.data.edge_index)
        recont_loss = recont_loss_cftf + recont_loss_cftt

        # disagreement regularization
        cosine_loss_f = F.cosine_similarity(E['pos_z_f'], E['pos_z_cf']).mean()
        cosine_loss_t = F.cosine_similarity(E['pos_z_t'], E['pos_z_ct']).mean()
        cosine_loss = -(cosine_loss_f + cosine_loss_t)
        assert not torch.isnan(mi_loss_f).any(), "mi_loss_f contains NaN values"
        assert not torch.isnan(mi_loss_t).any(), "mi_loss_t contains NaN values"
        assert not torch.isnan(mi_loss_cf).any(), "mi_loss_cf contains NaN values"
        assert not torch.isnan(recont_loss).any(), "recont_loss contains NaN values"
        assert not torch.isnan(cosine_loss).any(), "cosine_loss contains NaN values"


        return -(mi_loss_f + mi_loss_t + self.common_representation_regularization *(mi_loss_cf - recont_loss) + self.disagreement_regularization * cosine_loss)
        
    def get_downstream_embeddings(self):
        E = self.__compute_embeddings()
        return torch.stack([E['pos_z_f'], E['pos_z_t'], E['pos_z_cft']], dim=-1).mean(dim=-1)
