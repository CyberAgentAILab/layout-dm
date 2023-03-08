import argparse
import copy
import logging
import math
from functools import partial
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.mask import sample_mask
from trainer.helpers.sampling import sample
from trainer.helpers.task import duplicate_cond
from trainer.helpers.util import batch_topk_mask
from trainer.models.base_model import BaseModel
from trainer.models.common.nn_lib import (
    CategoricalTransformer,
    CustomDataParallel,
    SeqLengthDistribution,
)
from trainer.models.common.util import get_dim_model

logger = logging.getLogger(__name__)


# https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py
def mask_schedule_func(
    ratio: float, schedule: str, total_unknown: Optional[int] = None
):
    """Generates a mask rate by scheduling mask functions R.
    Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
    training, the input ratio is uniformly sampled; during inference, the input
    ratio is based on the step number divided by the total iteration number: t/T.
    Based on experiements, we find that masking more in training helps.
    Args:
      ratio: The uniformly sampled ratio [0, 1) as input.
      total_unknown: The total number of tokens that can be masked out. For
        example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
        512x512 images.
      method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
        "pow2.5" represents x^2.5
    Returns:
      The mask rate (float).
    """
    assert 0.0 <= torch.min(ratio) and torch.max(ratio) <= 1.0
    exp_dict = {"square": 2, "cubic": 3, "sqrt": 0.5}

    if total_unknown:
        total_unknown = torch.full(ratio.size(), total_unknown)

    if schedule == "linear":
        mask_ratio = 1.0 - ratio
    elif schedule == "cosine":
        mask_ratio = torch.cos(math.pi * 0.5 * ratio)
    elif schedule in exp_dict:
        mask_ratio = 1.0 - torch.pow(ratio, exp_dict[schedule])
    elif schedule == "log":
        mask_ratio = -1.0 * torch.log2(ratio) / torch.log2(total_unknown)
    elif schedule == "exp":
        mask_ratio = 1.0 - torch.exp2(-1.0 * torch.log2(total_unknown) * (1 - ratio))
    else:
        raise NotImplementedError

    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.0)
    return mask_ratio


class Wrapper(nn.Module):
    def __init__(self, generator: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.generator = generator
        self.critic = critic

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)


class MaskGIT(BaseModel):
    """
    To reproduce
    MaskGIT: Masked Generative Image Transformer (CVPR2022)
    https://arxiv.org/abs/2202.04200
    """

    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        mask_schedule: str = "cosine",
        use_token_critic: bool = False,
        use_padding_as_vocab: bool = False,
        use_gumbel_noise: bool = True,
    ) -> None:
        super().__init__()

        if use_padding_as_vocab:
            assert tokenizer.pad_until_max

        self.tokenizer = tokenizer
        self.mask_schedule = mask_schedule
        self.use_padding_as_vocab = use_padding_as_vocab
        self.use_gumbel_noise = use_gumbel_noise

        self.mask_schedule_func = partial(mask_schedule_func, schedule=mask_schedule)

        backbone = instantiate(backbone_cfg)
        self.model = CustomDataParallel(
            CategoricalTransformer(
                backbone=backbone,
                dim_model=get_dim_model(backbone_cfg),
                num_classes=self.tokenizer.N_total,
                max_token_length=tokenizer.max_token_length,
            )
        )

        # Note: make sure learnable parameters are inside self.model
        self.apply(self._init_weights)
        self.compute_stats()
        self.seq_dist = SeqLengthDistribution(tokenizer.max_seq_length)
        self.loss_fn_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_fn_bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        loss_mask = inputs["loss_mask"]

        if self.use_padding_as_vocab:
            outputs = self.model(inputs["input"])
        else:
            outputs = self.model(
                inputs["input"], src_key_padding_mask=inputs["padding_mask"]
            )
        nll_loss = self.loss_fn_ce(
            outputs["logits"][loss_mask],
            inputs["target"][loss_mask],
        )
        losses = {"nll_loss": nll_loss}

        # replace masked tokens with predicted tokens
        outputs["outputs"] = copy.deepcopy(inputs["input"])
        ids = torch.argmax(outputs["logits"], dim=-1)
        outputs["outputs"][loss_mask] = ids[loss_mask]
        return outputs, losses

    def sample(
        self,
        batch_size: Optional[int],
        cond: Optional[Tensor] = None,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Generate sample based on z.
        z can be either given or sampled from some distribution (e.g., normal)
        """

        mask_id = self.tokenizer.name_to_id("mask")
        pad_id = self.tokenizer.name_to_id("pad")
        B, S = (batch_size, self.tokenizer.max_token_length)
        T = sampling_cfg.get("num_timesteps", 10)

        if cond:
            cond = duplicate_cond(cond, batch_size)
            seq = cond["seq"].clone()
            # **_user will not be updated (kept as reference)
            seq_user = cond["seq"].clone()
            mask_user = cond["mask"].clone()
            if not self.use_padding_as_vocab:
                src_key_padding_mask_user = seq == pad_id
        else:
            n_elements = self.seq_dist.sample(B) * self.tokenizer.N_var_per_element
            indices = rearrange(torch.arange(S), "s -> 1 s")
            mask = indices < rearrange(n_elements, "b -> b 1")
            seq = torch.full((B, S), fill_value=pad_id)
            seq[mask] = mask_id
            seq_user = seq.clone()
            mask_user = ~mask.clone()
            src_key_padding_mask_user = ~mask.clone()

        if cond is None or cond["type"] != "partial":
            is_element_num_known = True
            element_mask = seq != pad_id
        else:
            is_element_num_known = False

        for t in range(T):
            float_t = torch.full((B,), (t + 1) / T)  # 1/T -> 1.0
            mask_ratio = self.mask_schedule_func(float_t)  # 1.0 -> 0.0
            temperature_at_t = sampling_cfg.temperature * (1.0 - float_t)
            is_masked = seq == mask_id

            with torch.no_grad():
                if self.use_padding_as_vocab:
                    out = self.model(seq.to(self.device))
                else:
                    out = self.model(
                        seq.to(self.device),
                        src_key_padding_mask=src_key_padding_mask_user.to(self.device),
                    )
            logits = out["logits"].cpu()

            invalid = repeat(~self.tokenizer.token_mask, "s c -> b s c", b=B)
            if is_element_num_known:
                # avoid predicting [PAD]
                X = self.tokenizer.N_total
                pad_mask = repeat(element_mask, "b s -> b s x", x=X)
                pad_mask = pad_mask & (
                    rearrange(torch.arange(X), "x -> 1 1 x") == pad_id
                )
                invalid = invalid | pad_mask
            logits[invalid] = -float("Inf")

            seq_pred = sample(rearrange(logits, "b s c -> b c s"), sampling_cfg)
            seq_pred = rearrange(seq_pred, "b 1 s -> b s")

            probs = F.softmax(logits, dim=2)
            confidence = torch.gather(
                torch.log(probs),
                2,
                rearrange(seq_pred, "b s -> b s 1"),
            )
            confidence = rearrange(confidence, "b s 1 -> b s")
            if self.use_gumbel_noise:
                # add gumbel noise in choosing tokens
                # https://github.com/google-research/maskgit/blob/cf615d448642942ddebaa7af1d1ed06a05720a91/maskgit/libml/parallel_decode.py#L29
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(confidence) + 1e-30) + 1e-30
                )
                # larger temp. adds more randomness
                confidence += rearrange(temperature_at_t, "b -> b 1") * gumbel_noise
            # non-masked region is kept forever
            # confidence[~is_masked] = CONFIDENCE_OF_KNOWN
            seq = torch.where(is_masked, seq_pred, seq)

            if t < T - 1:
                # re-fill [MASK] for unconfident predictions
                n_elem = reduce(~mask_user, "b s -> b", reduction="sum")
                topk = (n_elem * mask_ratio).long()
                is_unconfident, _ = batch_topk_mask(
                    -1.0 * confidence, topk, mask=is_masked
                )
                seq[is_unconfident] = mask_id

            # make sure to use user-defined inputs
            seq[mask_user] = seq_user[mask_user]

        layouts = self.tokenizer.decode(seq)
        return layouts

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: argparse.Namespace
    ) -> DictConfig:
        # MaskGIT follows original sampling scheme, so fix it regardless of the input.
        assert sampling_cfg.name == "random"
        assert sampling_cfg.temperature == 1.0
        sampling_cfg = super().aggregate_sampling_settings(sampling_cfg, args)
        return sampling_cfg

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        self.seq_dist(mask)

        inputs = self.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        B = inputs["seq"].size(0)
        mask_id = self.tokenizer.name_to_id("mask")

        mask_ratio = self.mask_schedule_func(ratio=torch.rand((B,)))
        if self.use_padding_as_vocab:
            loss_mask = sample_mask(torch.full(inputs["mask"].size(), True), mask_ratio)
        else:
            loss_mask = sample_mask(inputs["mask"], mask_ratio)

        masked_seq = copy.deepcopy(inputs["seq"])
        masked_seq[loss_mask] = mask_id

        return {
            "target": inputs["seq"],
            "padding_mask": ~inputs["mask"],
            "loss_mask": loss_mask,
            "input": masked_seq,
        }

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        additional_no_decay = [
            "model.module.pos_emb.pos_emb",
        ]

        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt_1 = plt.figure(figsize=(4, 4))
    x = torch.linspace(0.0, 1.0, 100)
    methods = ["linear", "cosine", "square", "cubic", "sqrt", "log", "exp"]
    for method in methods:
        y = mask_schedule_func(x, method=method, total_unknown=256)
        plt.plot(x, y, label=method)
    plt.legend()
    plt.savefig("mask_schedule.pdf")
