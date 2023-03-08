import copy
import logging
import pickle
from typing import Dict, List, Union

import fsspec
import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from torch import BoolTensor, FloatTensor, LongTensor
from trainer.global_configs import KMEANS_WEIGHT_ROOT
from trainer.helpers.clustering import Percentile

logger = logging.getLogger(__name__)

KEY_MULT_DICT = {
    "x-y-w-h": {"y": 1, "w": 2, "h": 3},
    "xywh": {},
}


class DummyClusteringModel:
    def __init__(self, cluster_centers: np.ndarray):
        self.cluster_centers_ = cluster_centers


class BboxTokenizer:
    """
    If N is number of bins, 0 <= x, y <= (N - 1) / N and 1 / N <= w, h <= 1
    'bbox' variable is assumed to have "xywh" order
    """

    def __init__(
        self,
        num_bin_bboxes: int,
        var_order: str = "c-x-y-w-h",
        shared_bbox_vocab: str = "xywh",
        bbox_quantization: str = "linear",
        dataset_name: str = "rico25_max25",
    ):
        # if bbox_quantization == "kmeans":
        #     assert shared_bbox_vocab == "x-y-w-h"

        self._num_bin_bboxes = num_bin_bboxes
        self._var_order = var_order.lstrip("c-").split("-")
        self._shared_bbox_vocab = shared_bbox_vocab
        self._bbox_quantization = bbox_quantization
        self._dataset_name = dataset_name
        self._var_names = ["x", "y", "w", "h"]

        self._clustering_models = {}
        if self.bbox_quantization in ["kmeans", "percentile"]:
            name = f"{dataset_name}_{self.bbox_quantization}_train_clusters.pkl"
            path = f"{KMEANS_WEIGHT_ROOT}/{name}"
            with fsspec.open(path, "rb") as f:
                valid_keys = [f"{k}-{self.num_bin_bboxes}" for k in self._var_names]
                for key, model in pickle.load(f).items():
                    if key not in valid_keys:
                        continue

                    # sort cluster center in 1d case
                    var_name = key.split("-")[0]
                    if len(var_name) == 1:
                        cluster_centers = np.sort(
                            model.cluster_centers_, axis=0
                        )  # (N, 1)
                        model.cluster_centers_ = cluster_centers

                    self._clustering_models[key] = model
        else:
            for n in self.var_names:
                d = 1 / self.num_bin_bboxes
                if n in ["x", "y", "z"]:
                    centers = np.linspace(
                        start=0.0, stop=1.0 - d, num=self.num_bin_bboxes
                    )
                else:
                    centers = np.linspace(start=d, stop=1.0, num=self.num_bin_bboxes)
                centers = rearrange(centers, "c -> c 1")
                key = f"{n}-{self.num_bin_bboxes}"
                self._clustering_models[key] = DummyClusteringModel(centers)

    def encode(self, bbox: FloatTensor) -> LongTensor:
        d = 1 / self.num_bin_bboxes  # delta
        bbox_q = torch.zeros_like(bbox)

        if self.bbox_quantization == "linear":
            bbox_q[..., :2] = torch.clamp(bbox[..., :2], 0.0, 1.0 - d)  # ["x", "y"]
            bbox_q[..., 2:] = torch.clamp(bbox[..., 2:], d, 1.0) - d  # ["w", "h"]
            indices = (self.num_bin_bboxes * bbox_q).round().long()

        elif self.bbox_quantization in ["kmeans", "percentile"]:
            B, S = bbox.size()[:2]
            indices = []
            if len(self._var_order) == 4:
                for i, key in enumerate(self._var_names):
                    model = self.clustering_models[f"{key}-{self.num_bin_bboxes}"]
                    input_ = bbox[..., i : i + 1].view(-1, 1).numpy().astype(np.float32)
                    output = torch.from_numpy(model.predict(input_)).view(B, S, 1)
                    indices.append(output)

            indices = torch.cat(indices, dim=-1)

        # add offset if vocabularies are not fully shared among xywh
        if len(self._var_order) == 4:
            for (key, mult) in KEY_MULT_DICT[self.shared_bbox_vocab].items():
                indices[..., self._var_names.index(key)] += self.num_bin_bboxes * mult

        # change var order if necessary
        if len(self._var_order) == 4:
            order_indices = [self._var_names.index(k) for k in self._var_order]
            indices = indices[..., order_indices]

        return indices

    def decode(self, bbox_indices: LongTensor) -> FloatTensor:
        arr = torch.clone(bbox_indices)  # avoid overriding

        # restore var order back to "xywh" if necessary
        if len(self._var_order) == 4:
            order_indices = [self._var_order.index(k) for k in self._var_names]
            arr = arr[..., order_indices]

        # subtract offset if vocabularies are not fully shared among xywh
        if len(self._var_order) == 4:
            for (key, mult) in KEY_MULT_DICT[self.shared_bbox_vocab].items():
                arr[..., self._var_names.index(key)] -= self.num_bin_bboxes * mult

        if self.bbox_quantization == "linear":
            # if len(self._var_order) == 2:
            #     # decode from product space
            #     tmp = {}
            #     for i, vs in enumerate(self._var_order):
            #         x = torch.clone(arr[..., i])
            #         for v in reversed(vs):
            #             tmp[v] = x % self.num_bin_bboxes
            #             x = torch.div(x, self.num_bin_bboxes, rounding_mode="floor")
            #     arr = torch.stack([tmp[k] for k in self.keys], dim=-1)

            arr = torch.clamp(arr, 0, self.num_bin_bboxes - 1)  # avoid OOV

            bbox = torch.zeros(arr.size()).float()
            d = 1 / self.num_bin_bboxes
            bbox[..., :2] = arr[..., :2].float() * d
            bbox[..., 2:] = (arr[..., 2:] + 1).float() * d

        elif self.bbox_quantization in ["kmeans", "percentile"]:
            B, S = arr.size()[:2]
            arr = torch.clamp(arr, 0, self.num_bin_bboxes - 1)  # avoid OOV
            bbox = []
            if len(self._var_order) == 4:
                for i, key in enumerate(self._var_names):
                    model = self.clustering_models[f"{key}-{self.num_bin_bboxes}"]
                    inds = arr[..., i : i + 1].view(-1).numpy()
                    loc = torch.from_numpy(model.cluster_centers_[inds]).view(B, S, -1)
                    bbox.append(loc)
            elif len(self._var_order) == 2:
                tmp = {}
                for i, vs in enumerate(self._var_order):
                    model = self.clustering_models[f"{vs}-{self.num_bin_bboxes}"]
                    inds = arr[..., i : i + 1].view(-1).numpy()
                    inds = model.cluster_centers_[inds]
                    loc = torch.from_numpy(inds).view(B, S, -1)
                    for j, key in enumerate(vs):
                        tmp[key] = loc[..., j : j + 1]

                # reorganize var order to xywh
                bbox = [tmp[key] for key in self._var_names]

            bbox = torch.cat(bbox, dim=-1)
            bbox = torch.clamp(bbox, 0.0, 1.0)

        return bbox

    @property
    def bbox_vocab_len(self) -> int:
        return self.num_bin_bboxes * len(self.shared_bbox_vocab.split("-"))

    @property
    def bbox_quantization(self) -> str:
        return self._bbox_quantization

    @property
    def clustering_models(
        self,
    ) -> Dict[str, Union[KMeans, Percentile, DummyClusteringModel]]:
        return self._clustering_models

    @property
    def num_bin_bboxes(self) -> int:
        return self._num_bin_bboxes

    @property
    def shared_bbox_vocab(self) -> str:
        return self._shared_bbox_vocab

    @property
    def token_mask(self) -> Dict[str, BoolTensor]:
        masks = {}
        if self.shared_bbox_vocab == "xywh":
            for key in self._var_order:
                masks[key] = torch.full((self.num_bin_bboxes,), True)
        elif self.shared_bbox_vocab == "x-y-w-h":
            key_mult = KEY_MULT_DICT["x-y-w-h"]
            S = self.num_bin_bboxes * 4
            false_tensor = torch.full((S,), False)
            for key in self._var_order:
                masks[key] = copy.deepcopy(false_tensor)
                i = key_mult.get(key, 0)
                start, stop = i * self.num_bin_bboxes, (i + 1) * self.num_bin_bboxes
                masks[key][start:stop] = True
        else:
            raise NotImplementedError

        return masks

    @property
    def var_names(self) -> List[str]:
        return self._var_names
