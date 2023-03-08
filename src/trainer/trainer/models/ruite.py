import logging
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.utils import to_dense_batch
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import sample
from trainer.helpers.task import duplicate_cond
from trainer.models.base_model import BaseModel
from trainer.models.common.nn_lib import CategoricalTransformer, CustomDataParallel
from trainer.models.common.util import get_dim_model

logger = logging.getLogger(__name__)


class RUITE(BaseModel):
    """
    To reproduce
    Refining ui layout aesthetics using transformer encoder
    https://dl.acm.org/doi/10.1145/3397482.3450716

    """

    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        backbone = instantiate(backbone_cfg)
        self.model = CustomDataParallel(
            CategoricalTransformer(
                backbone=backbone,
                dim_model=get_dim_model(backbone_cfg),
                num_classes=self.tokenizer.N_total,
                max_token_length=tokenizer.max_token_length,
                lookahead=True,
            )
        )

        # Note: make sure learnable parameters are inside self.model
        self.apply(self._init_weights)
        self.compute_stats()
        self.loss_fn_ce = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.name_to_id("pad")
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        outputs = self.model(
            inputs["input"], src_key_padding_mask=inputs["padding_mask"]
        )
        nll_loss = self.loss_fn_ce(
            rearrange(outputs["logits"], "b s c -> b c s"), inputs["target"]
        )

        losses = {"nll_loss": nll_loss}
        outputs["outputs"] = torch.argmax(outputs["logits"], dim=-1)
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
        pad_id = self.tokenizer.name_to_id("pad")

        if cond:
            cond = duplicate_cond(cond, batch_size)
            padding_mask = cond["seq"] == pad_id
            outputs = self.model(
                cond["seq"].to(self.device),
                src_key_padding_mask=padding_mask.to(self.device),
            )
            logits = rearrange(outputs["logits"].cpu(), "b s c -> b c s")
            seq = rearrange(sample(logits, sampling_cfg), "b 1 s -> b s")
            seq[cond["mask"]] = cond["seq"][cond["mask"]]
        else:
            # since RUITE cannot generate without inputs, just generate dummy
            seq = torch.full(
                (batch_size, self.tokenizer.max_token_length), fill_value=pad_id
            )
            seq[:, 0] = 0
            seq[:, 1:5] = self.tokenizer.N_category

        layouts = self.tokenizer.decode(seq)
        return layouts

    def preprocess(self, batch):
        bbox_w_noise, label, _, mask = sparse_to_dense(batch)
        bbox, _ = to_dense_batch(batch.x_orig, batch.batch)
        inputs = self.tokenizer.encode(
            {"label": label, "mask": mask, "bbox": bbox_w_noise}
        )
        targets = self.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})

        return {
            "target": targets["seq"],
            "padding_mask": ~inputs["mask"],
            "input": inputs["seq"],
        }

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        return super().optim_groups(
            weight_decay=weight_decay,
            additional_no_decay=[
                "model.module.pos_emb.pos_emb",
            ],
        )
