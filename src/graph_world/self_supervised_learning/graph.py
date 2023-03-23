from typing import Dict, List, Union
from torch_geometric.data.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
import torch


class NodeMappings:
    '''
    Maps node indices between a source graph to a target graph. The k-th smallest node in the source graph is mapped to node k in the target graph.
    '''
    def __init__(self, src_nodes : Union[torch.Tensor, List]):
        if isinstance(src_nodes, torch.Tensor):
            assert src_nodes.dim() == 1
            src_nodes = src_nodes.detach().numpy().tolist()
        src_nodes_sorted = sorted(src_nodes)

        self.__src_to_target = { source: target for (target, source) in enumerate(src_nodes_sorted) }
        self.__target_to_src = src_nodes_sorted
    
    def src_to_target(self, src_node: int) -> int:
        return self.__src_to_target[src_node]
    
    def target_to_src(self, target_node: int) -> int:
        return self.__target_to_src[target_node]
    
    @property
    def all_target_to_src(self) -> List[int]:
        return self.__target_to_src
    
    @property
    def num_nodes(self) -> int:
        return len(self.__target_to_src)

    


class SubGraph:
    '''
    Create a subgraph G' from graph G. Note this does not preserve the edge attributes.
    '''
    def __init__(self, node_indices: Union[torch.Tensor, List], data: Data, subgraph_edges = None, **kwargs):
        '''
        args
        ----
        node_indices: 
            Indices of the original graph G
        data:
            Data object of the original graph G
        subgraph_edges:
            Edges that are to be includes in the subgraph. If this is None, it will be computed using the subgraph method in
            pytorch geometric from the node_indices.
        '''
        self.node_mapping = NodeMappings(src_nodes=node_indices)
        self.__subgraph = None
        self.__full_graph = data
        self.__node_indices = node_indices
        self.__subgraph_edges_cached = subgraph_edges

    @property
    def subgraph_number_of_nodes(self):
        return len(self.__node_indices)
    
    @property
    def __subgraph_edges(self):
        if self.__subgraph_edges_cached is None:
            self.__subgraph_edges_cached , *_ = subgraph(self.__node_indices, edge_index=self.__full_graph.edge_index, relabel_nodes=True, num_nodes = self.__full_graph.num_nodes)
        return self.__subgraph_edges_cached

    @property
    def subgraph_data(self) -> Data:
        if self.__subgraph is None:
            subgraph_x = self.__full_graph.x[self.node_mapping.all_target_to_src, :]
            subgraph_y = self.__full_graph.y[self.node_mapping.all_target_to_src]
            self.__subgraph = Data(x=subgraph_x, edge_index=self.__subgraph_edges, y=subgraph_y)

        return self.__subgraph

    @property
    def original_graph_data(self) -> Data:
        return self.__full_graph


class SubGraphs:
    def __init__(self, subgraphs: List[SubGraph]):
        self.__subgraph_data_list = subgraphs
        self.subgraph_data = Batch.from_data_list(
            [subgraph.subgraph_data for subgraph in subgraphs]
        )

        # Subgraph offsets in the feature / embedding matrix of all merged subgraphs
        self.subgraph_offsets = [0] * (len(subgraphs) + 1)
        for (i, subgraph) in enumerate(subgraphs):
            self.subgraph_offsets[i + 1] = self.subgraph_offsets[i] + \
                subgraph.subgraph_number_of_nodes
            
    @property
    def subgraph_data_list(self) -> List[SubGraph]:
        return self.__subgraph_data_list
    
    @property
    def subgraph_batches(self):
        return self.subgraph_data

    @property
    def n_subgraphs(self) -> int:
        return len(self.subgraph_data_list)
    
    def get_subgraph_offset(self, subgraph_idx : int) -> int:
        return self.subgraph_offsets[subgraph_idx]

    def get_subgraph(self, subgraph_idx : int) -> SubGraph:
        return self.subgraph_data_list[subgraph_idx]
    
