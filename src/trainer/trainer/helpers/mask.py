from typing import Optional, Union

import torch
from einops import rearrange, reduce, repeat
from torch import BoolTensor, FloatTensor, LongTensor

from .util import batch_topk_mask


def sequence_mask(length: LongTensor, maxlen: Optional[int] = None) -> BoolTensor:
    """
    Similar to https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    """
    B = length.size(0)
    maxlen = maxlen if maxlen else length.max()
    indices = repeat(torch.arange(maxlen), "s -> b s", b=B)
    mask = indices < rearrange(length, "b -> b 1")
    return mask


def sample_mask(mask: BoolTensor, ratio: Union[float, FloatTensor]) -> BoolTensor:
    """
    Generate sampled_mask (B, S) given mask (B, S) according to the specified ratio
    If mask[b, s] is False, sampled_mask[b, s] should be False.
    """
    if isinstance(ratio, float):
        ratio = torch.full((mask.size(0),), fill_value=ratio)

    scores = torch.rand(mask.size())
    n_elem = reduce(mask, "b s -> b", reduction="sum")
    topk = (ratio * n_elem).long()
    sampled_mask, _ = batch_topk_mask(scores, topk, mask=mask)
    return sampled_mask


if __name__ == "__main__":
    sample_mask(torch.full((2, 3), fill_value=False), 0.5)
