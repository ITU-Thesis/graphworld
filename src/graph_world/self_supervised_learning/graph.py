from typing import Dict, List, Union
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import torch

class SubGraph:
    '''
    Create a subgraph G' from graph G. This keeps track of the indices from the nodes in the subgraph G' (new indices)
    to the nodes in the graph G (old indices).
    Note this does not preserve the edge attributes.
    '''
    def __init__(self, node_indices: Union[torch.Tensor, List], data: Data, **kwargs):
        if isinstance(node_indices, torch.Tensor):
            assert node_indices.dim() == 1
            node_indices = node_indices.detach().numpy().tolist()
        
        self.__subgraph = None
        self.__full_graph = data
        self.__node_indices = node_indices
        self.__new_to_old_node_indices = None
        self.__old_to_new_node_indices = None

    @property
    def subgraph_number_of_nodes(self):
        return len(self.__node_indices)

    @property
    def new_to_old_node_indices(self) -> List[int]:
        '''
        Map the new indices (from the subgraph) to the old indices (from the original graph)
        Index 0 maps to the smallest old index.
        '''
        if self.__new_to_old_node_indices is None:
            self.__new_to_old_node_indices = sorted(self.__node_indices)
        return self.__new_to_old_node_indices
    
    @property
    def old_to_new_node_indices(self) -> Dict[int, int]:
        '''
        Map the old indices (from the original graph) to the new indices (in the subgraph).
        Smallest old index maps to index 0.
        '''
        if self.__old_to_new_node_indices is None:
            self.__old_to_new_node_indices = { old_index: new_index for new_index, old_index in enumerate(self.new_to_old_node_indices) }
        return self.__old_to_new_node_indices

    @property
    def subgraph_data(self) -> Data:
        if self.__subgraph is None:
            subgraph_edges, *_ = subgraph(self.__node_indices, edge_index=self.__full_graph.edge_index, relabel_nodes=True)
            subgraph_x = self.__full_graph.x[self.new_to_old_node_indices, :]
            subgraph_y = self.__full_graph.y[self.new_to_old_node_indices]
            self.__subgraph = Data(x=subgraph_x, edge_index=subgraph_edges, y=subgraph_y)

        return self.__subgraph

    @property
    def original_graph_data(self) -> Data:
        return self.__full_graph
    
    def get_old_to_new_index(self, old_index : int) -> int:
        assert old_index in self.old_to_new_node_indices
        return self.old_to_new_node_indices[old_index]
    
    def get_new_to_old_index(self, new_index : int) -> int:
        assert new_index in self.new_to_old_node_indices
        return self.new_to_old_node_indices[new_index]


