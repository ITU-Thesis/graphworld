from torch_geometric.data import Data, Dataset
from typing import Union, Dict

# class EvaluationMetrics(TypedDict):
#     accuracy: float
#     f1_micro: float
#     f1_macro: float
#     rocauc_ovr: float
#     rocauc_ovr: float
#     logloss: float

EvaluationMetrics = Dict[str, float]
    

InputGraph = Union[Data, Dataset]