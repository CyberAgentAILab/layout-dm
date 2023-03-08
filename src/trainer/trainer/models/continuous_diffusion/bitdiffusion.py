from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import FloatTensor, LongTensor
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer

from .base import ContinuousDiffusionBase


def ids_to_bits(x: LongTensor, num_bits: int) -> FloatTensor:
    """
    Given ids with shape (B, S), returns bits (in -1. or 1. form) with shape (B, S, bits)
    """
    assert x.max().item() < 2**num_bits
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(x.device)
    mask = rearrange(mask, "d -> 1 d")
    x = rearrange(x, "b s -> b s 1")

    bits = ((x & mask) != 0).float()
    bits = bits * 2 - 1.0
    return bits


def bits_to_ids(
    x: FloatTensor, num_bits: int, tokenizer: Optional[LayoutSequenceTokenizer] = None
) -> LongTensor:
    B, S, _ = x.size()
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, dtype=torch.int32).to(x.device)
    mask = rearrange(mask, "d -> 1 d")

    bits = (x > 0).int()
    if tokenizer:
        base_ids = rearrange(torch.arange(2**num_bits), "d -> 1 d")
        base_bits = rearrange(ids_to_bits(base_ids, num_bits), "1 n c -> 1 1 n c")
        dist = torch.abs(rearrange(x, "b s c -> b s 1 c") - base_bits.to(x.device))
        dist = reduce(dist, "b s n c -> b s n", reduction="sum")

        pad = torch.full((S, 2**num_bits - tokenizer.N_total), False)
        valid = torch.cat([tokenizer.token_mask, pad], dim=1)
        valid = repeat(valid, "s x -> b s x", b=B).to(x.device)
        dist = torch.where(valid, dist, torch.full_like(dist, fill_value=float("Inf")))
        ids = torch.argmax(-dist, dim=-1)
    else:
        ids = reduce(bits * mask, "b s d -> b s", "sum").long()
    return ids


class BitDiffusion(ContinuousDiffusionBase):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scale = 1.0
        self.con2logits = None

    def dis2con(
        self, seq: LongTensor, reparametrize: bool = False, normalize: bool = False
    ) -> Tuple[FloatTensor, FloatTensor]:
        assert seq.dim() == 2
        # return ids_to_bits(seq, self.num_channel) * self.scale, None
        x = ids_to_bits(seq, self.num_channel) * self.scale
        return x, x

    def con2dis(self, arr: FloatTensor) -> LongTensor:
        assert arr.dim() == 3
        return bits_to_ids(arr, self.num_channel, self.tokenizer)
