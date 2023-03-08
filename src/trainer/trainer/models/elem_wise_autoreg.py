import logging
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import sample
from trainer.helpers.task import (
    duplicate_cond,
    set_additional_conditions_for_refinement,
)
from trainer.models.base_model import BaseModel
from trainer.models.common.nn_lib import CategoricalTransformer, CustomDataParallel
from trainer.models.common.util import get_dim_model

logger = logging.getLogger(__name__)


class ElemWiseAutoreg(BaseModel):
    """
    To reproduce
    LayoutTransformer: Layout Generation and Completion with Self-attention (ICCV2021)
    https://arxiv.org/abs/2006.14615
    """

    def __init__(
        self,
        # cfg: DictConfig,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        pos_emb: str = "default",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        kwargs = {}
        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = tokenizer.N_var_per_element

        # Note: make sure learnable parameters are inside self.model
        backbone = instantiate(backbone_cfg)
        self.model = CustomDataParallel(
            CategoricalTransformer(
                backbone=backbone,
                dim_model=get_dim_model(backbone_cfg),
                num_classes=self.tokenizer.N_total,
                max_token_length=tokenizer.max_token_length + 1,  # +1 for BOS
                pos_emb=pos_emb,
                lookahead=False,
                **kwargs,
            )
        )
        self.apply(self._init_weights)
        self.compute_stats()

        self.loss_fn_ce = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=self.tokenizer.name_to_id("pad")
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        outputs = self.model(inputs["input"])
        nll_loss = self.loss_fn_ce(
            rearrange(outputs["logits"], "b s c -> b c s"),
            inputs["target"],
        )
        losses = {"nll_loss": nll_loss}
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

        if cond and cond["type"] == "refinement":
            # additional weak constraints
            cond = set_additional_conditions_for_refinement(
                cond, self.tokenizer, sampling_cfg
            )

        if cond:
            cond = duplicate_cond(cond, batch_size)
            for k, v in cond.items():
                if isinstance(v, Tensor):
                    cond[k] = v.to(self.device)

        special_keys = self.tokenizer._special_token_name_to_id
        mask_id = self.tokenizer.name_to_id("mask") if "mask" in special_keys else -1
        input_ = torch.full(
            (batch_size, 1), fill_value=self.tokenizer.name_to_id("bos")
        )
        input_ = input_.to(self.device)

        for i in range(self.tokenizer.max_token_length):
            with torch.no_grad():
                logits = self.model(input_)["logits"]
            logits = rearrange(logits[:, i : i + 1], "b 1 c -> b c")

            if cond:
                if cond.get("type", None) == "relation":
                    raise NotImplementedError
                    # using sparse DataBatch and partial dense array to compute relation is "practically" very complex,
                    # (i) add dummy logits to match the dimension (easy)
                    # (ii) add "causal" loss mask because interaction between i-th and i'-th (i' < i) is valid (hard)
                # impose weak user-specified constraints by addition
                if cond.get("type", None) == "refinement":
                    weak_mask = cond["weak_mask"][..., i + 1]
                    weak_logits = cond["weak_logits"][..., i + 1]
                    logits[weak_mask] += weak_logits[weak_mask]

            invalid = repeat(
                ~self.tokenizer.token_mask[i : i + 1], "1 c -> b c", b=input_.size(0)
            )
            logits[invalid] = -float("Inf")

            predicted = sample(logits, sampling_cfg)
            if cond:
                id_ = cond["seq"][:, i + 1 : i + 2]
                if id_.size(1) == 1:
                    # If condition exists and is valid, use it
                    flag = id_ == mask_id
                    predicted = torch.where(flag, predicted, id_)
            input_ = torch.cat([input_, predicted], dim=1)

        ids = input_[:, 1:].cpu()  # pop BOS
        layouts = self.tokenizer.decode(ids)
        return layouts

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        x = self.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        input_ = x["seq"][:, :-1]
        target = x["seq"][:, 1:]
        return {"input": input_, "target": target}

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        base = "model.module.pos_emb"
        additional_no_decay = [
            f"{base}.{name}" for name in self.model.module.pos_emb.no_decay_param_names
        ]
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
