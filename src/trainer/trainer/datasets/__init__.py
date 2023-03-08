from .publaynet import PubLayNetDataset
from .rico import Rico25Dataset

_DATASETS = [
    Rico25Dataset,
    PubLayNetDataset,
]
DATASETS = {d.name: d for d in _DATASETS}
