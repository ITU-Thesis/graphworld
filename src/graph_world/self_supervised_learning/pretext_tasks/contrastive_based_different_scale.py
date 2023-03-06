from torch_geometric.nn.models.deep_graph_infomax import DeepGraphInfomax
from torch import Tensor
import torch.nn
import gin
from .basic_pretext_task import BasicPretextTask


@gin.configurable
class DeepGraphInfomax(BasicPretextTask):
    '''
    Deep Graph Infomax proposed in Velickovic, Petar, et al. "Deep graph infomax." ICLR (Poster) 2.3 (2019): 4.
    '''
    def __init__(self, **kwargs):

        def summary_fn(h): return h.mean(dim=0)
        def corruption(h): return h[torch.randperm(h.size()[0]), :]

        self.dgi = DeepGraphInfomax(
            hidden_channell=self.encoder.out_channel,
            encoder=self.encoder,
            summary=summary_fn,
            corruption=corruption
        )

    def make_loss(self, embeddings: Tensor):
        pos_z, neg_z, summary = self.dgi(self.data)
        return self.dgi.loss(pos_z=pos_z, neg_z=neg_z, summary=summary)

