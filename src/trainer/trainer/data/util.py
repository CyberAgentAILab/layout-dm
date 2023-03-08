import random
from enum import IntEnum
from itertools import combinations, product
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch import BoolTensor, FloatTensor, LongTensor
from torch_geometric.utils import to_dense_batch
from trainer.helpers.util import convert_xywh_to_ltrb


class RelSize(IntEnum):
    UNKNOWN = 0
    SMALLER = 1
    EQUAL = 2
    LARGER = 3


class RelLoc(IntEnum):
    UNKNOWN = 4
    LEFT = 5
    TOP = 6
    RIGHT = 7
    BOTTOM = 8
    CENTER = 9


REL_SIZE_ALPHA = 0.1


def detect_size_relation(b1, b2):
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    alpha = REL_SIZE_ALPHA
    if (1 - alpha) * a1 < a2 < (1 + alpha) * a1:
        return RelSize.EQUAL
    elif a1 < a2:
        return RelSize.LARGER
    else:
        return RelSize.SMALLER


def detect_loc_relation(b1, b2, is_canvas=False):
    if is_canvas:
        yc = b2[1]
        if yc < 1.0 / 3:
            return RelLoc.TOP
        elif yc < 2.0 / 3:
            return RelLoc.CENTER
        else:
            return RelLoc.BOTTOM

    else:
        l1, t1, r1, b1 = convert_xywh_to_ltrb(b1)
        l2, t2, r2, b2 = convert_xywh_to_ltrb(b2)

        if b2 <= t1:
            return RelLoc.TOP
        elif b1 <= t2:
            return RelLoc.BOTTOM
        elif r2 <= l1:
            return RelLoc.LEFT
        elif r1 <= l2:
            return RelLoc.RIGHT
        else:
            # might not be necessary
            return RelLoc.CENTER


def get_rel_text(rel, canvas=False):
    if type(rel) == RelSize:
        index = rel - RelSize.UNKNOWN - 1
        if canvas:
            return [
                "within canvas",
                "spread over canvas",
                "out of canvas",
            ][index]

        else:
            return [
                "larger than",
                "equal to",
                "smaller than",
            ][index]

    else:
        index = rel - RelLoc.UNKNOWN - 1
        if canvas:
            return [
                "",
                "at top",
                "",
                "at bottom",
                "at middle",
            ][index]

        else:
            return [
                "right to",
                "below",
                "left to",
                "above",
                "around",
            ][index]


# transform
class AddCanvasElement:
    x = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float)
    y = torch.tensor([0], dtype=torch.long)

    def __call__(self, data):
        flag = data.attr["has_canvas_element"].any().item()
        assert not flag
        if not flag:
            # device = data.x.device
            # x, y = self.x.to(device), self.y.to(device)
            data.x = torch.cat([self.x, data.x], dim=0)
            data.y = torch.cat([self.y, data.y + 1], dim=0)
            data.attr = data.attr.copy()
            data.attr["has_canvas_element"] = True
        return data


class AddRelationConstraints:
    def __init__(self, seed=None, edge_ratio=0.1, use_v1=False):
        self.edge_ratio = edge_ratio
        self.use_v1 = use_v1
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)

    def __call__(self, data):
        N = data.x.size(0)
        has_canvas = data.attr["has_canvas_element"]

        rel_all = list(product(range(2), combinations(range(N), 2)))
        size = int(len(rel_all) * self.edge_ratio)
        rel_sample = set(self.generator.sample(rel_all, size))

        edge_index, edge_attr = [], []
        rel_unk = 1 << RelSize.UNKNOWN | 1 << RelLoc.UNKNOWN
        for i, j in combinations(range(N), 2):
            bi, bj = data.x[i], data.x[j]
            canvas = data.y[i] == 0 and has_canvas

            if self.use_v1:
                if (0, (i, j)) in rel_sample:
                    rel_size = 1 << detect_size_relation(bi, bj)
                    rel_loc = 1 << detect_loc_relation(bi, bj, canvas)
                else:
                    rel_size = 1 << RelSize.UNKNOWN
                    rel_loc = 1 << RelLoc.UNKNOWN
            else:
                if (0, (i, j)) in rel_sample:
                    rel_size = 1 << detect_size_relation(bi, bj)
                else:
                    rel_size = 1 << RelSize.UNKNOWN

                if (1, (i, j)) in rel_sample:
                    rel_loc = 1 << detect_loc_relation(bi, bj, canvas)
                else:
                    rel_loc = 1 << RelLoc.UNKNOWN

            rel = rel_size | rel_loc
            if rel != rel_unk:
                edge_index.append((i, j))
                edge_attr.append(rel)

        data.edge_index = torch.as_tensor(edge_index).long()
        data.edge_index = data.edge_index.t().contiguous()
        data.edge_attr = torch.as_tensor(edge_attr).long()

        return data


class RandomOrder:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        device = data.x.device
        N = data.x.size(0)
        idx = torch.randperm(N, device=device)
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class SortByLabel:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        idx = data.y.sort().indices
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class LexicographicOrder:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        x, y, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(y, x)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class AddNoiseToBBox:
    def __init__(self, std: float = 0.05):
        self.std = float(std)

    def __call__(self, data):
        noise = torch.normal(0, self.std, size=data.x.size(), device=data.x.device)
        data.x_orig = data.x.clone()
        data.x = data.x + noise
        data.attr = data.attr.copy()
        data.attr["NoiseAdded"][0] = True
        return data


class HorizontalFlip:
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 0] = 1 - data.x[:, 0]
        return data


# def compose_transform(transforms):
#     module = sys.modules[__name__]
#     transform_list = []
#     for t in transforms:
#         # parse args
#         if "(" in t and ")" in t:
#             args = t[t.index("(") + 1 : t.index(")")]
#             t = t[: t.index("(")]
#             regex = re.compile(r"\b(\w+)=(.*?)(?=\s\w+=\s*|$)")
#             args = dict(regex.findall(args))
#             for k in args:
#                 try:
#                     args[k] = float(args[k])
#                 except:
#                     pass
#         else:
#             args = {}
#         if isinstance(t, str):
#             if hasattr(module, t):
#                 transform_list.append(getattr(module, t)(**args))
#             else:
#                 raise NotImplementedError
#         else:
#             raise NotImplementedError
#     return T.Compose(transform_list)


def compose_transform(transforms: List[str]) -> T.Compose:
    """
    Compose transforms, optionally with args (e.g., AddRelationConstraints(edge_ratio=0.1))
    """
    transform_list = []
    for t in transforms:
        if "(" in t and ")" in t:
            pass
        else:
            t += "()"
        transform_list.append(eval(t))
    return T.Compose(transform_list)


def sparse_to_dense(
    batch,
    device: torch.device = torch.device("cpu"),
    remove_canvas: bool = False,
) -> Tuple[FloatTensor, LongTensor, BoolTensor, BoolTensor]:
    batch = batch.to(device)
    bbox, _ = to_dense_batch(batch.x, batch.batch)
    label, mask = to_dense_batch(batch.y, batch.batch)

    if remove_canvas:
        bbox = bbox[:, 1:].contiguous()
        label = label[:, 1:].contiguous() - 1  # cancel +1 effect in transform
        label = label.clamp(min=0)
        mask = mask[:, 1:].contiguous()

    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def loader_to_list(
    loader: torch.utils.data.dataloader.DataLoader,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    layouts = []
    for batch in loader:
        bbox, label, _, mask = sparse_to_dense(batch)
        for i in range(len(label)):
            valid = mask[i].numpy()
            layouts.append((bbox[i].numpy()[valid], label[i].numpy()[valid]))
    return layouts


def split_num_samples(N: int, batch_size: int) -> List[int]:
    quontinent = N // batch_size
    remainder = N % batch_size
    dataloader = quontinent * [batch_size]
    if remainder > 0:
        dataloader.append(remainder)
    return dataloader
