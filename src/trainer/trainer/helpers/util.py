import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch import BoolTensor, FloatTensor, LongTensor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def batch_topk_mask(
    scores: FloatTensor,
    topk: LongTensor,
    mask: Optional[BoolTensor] = None,
) -> Tuple[BoolTensor, FloatTensor]:
    assert scores.ndim == 2 and topk.ndim == 1 and scores.size(0) == topk.size(0)
    if mask is not None:
        assert mask.size() == scores.size()
        assert (scores.size(1) >= topk).all()

    # ignore scores where mask = False by setting extreme values
    if mask is not None:
        const = -1.0 * float("Inf")
        const = torch.full_like(scores, fill_value=const)
        scores = torch.where(mask, scores, const)

    sorted_values, _ = torch.sort(scores, dim=-1, descending=True)
    topk = rearrange(topk, "b -> b 1")

    k_th_scores = torch.gather(sorted_values, dim=1, index=topk)

    topk_mask = scores > k_th_scores
    return topk_mask, k_th_scores


def batch_shuffle_index(
    batch_size: int,
    feature_length: int,
    mask: Optional[BoolTensor] = None,
) -> LongTensor:
    """
    Note: masked part may be shuffled because of unpredictable behaviour of sorting [inf, ..., inf]
    """
    if mask:
        assert mask.size() == [batch_size, feature_length]
    scores = torch.rand((batch_size, feature_length))
    if mask:
        scores[~mask] = float("Inf")
    _, indices = torch.sort(scores, dim=1)
    return indices


if __name__ == "__main__":
    scores = torch.arange(6).view(2, 3).float()
    # topk = torch.arange(2) + 1
    topk = torch.full((2,), 3)
    mask = torch.full((2, 3), False)
    # mask[1, 2] = False
    print(batch_topk_mask(scores, topk, mask=mask))
