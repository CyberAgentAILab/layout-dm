import logging
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

logger = logging.getLogger(__name__)


class BboxEncoder(torch.nn.Module):
    def __init__(
        self,
        num_bin_bboxes: int,
        output_dim: int,
        fusion: str = "emb_concat",
    ) -> None:
        super().__init__()
        self.fusion = fusion
        if fusion == "linear":
            # self.emb = nn.Linear(4, output_dim)
            raise NotImplementedError
        elif fusion in ["emb_concat", "emb_add"]:
            self.x_emb = nn.Embedding(num_bin_bboxes, output_dim)
            self.y_emb = nn.Embedding(num_bin_bboxes, output_dim)
            self.w_emb = nn.Embedding(num_bin_bboxes, output_dim)
            self.h_emb = nn.Embedding(num_bin_bboxes, output_dim)
        else:
            raise NotImplementedError

    def forward(self, bbox: LongTensor) -> FloatTensor:
        if self.fusion == "linear":
            emb = self.emb(bbox.float())
        elif self.fusion in ["emb_concat", "emb_add"]:
            embs = []
            for (key, value) in zip(
                ["x", "y", "w", "h"],
                torch.split(bbox, split_size_or_sections=1, dim=-1),
            ):
                embs.append(
                    getattr(self, f"{key}_emb")(rearrange(value, "b s 1 -> b s"))
                )
            if self.fusion == "emb_add":
                emb = sum(embs)
            else:
                emb = torch.cat(embs, dim=-1)
        else:
            raise NotImplementedError
        return emb


class LayoutEncoder(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        num_classes: int,
        lb_fusion: str = "concat_fc",
    ) -> None:
        super().__init__()
        assert lb_fusion in ["add", "concat_fc"]
        self.lb_fusion = lb_fusion
        self.bbox_fusion = "emb_concat"

        if self.lb_fusion == "concat_fc":
            self.label_emb = nn.Embedding(num_classes, output_dim)
            self.bbox_emb = BboxEncoder(
                num_classes, output_dim, fusion=self.bbox_fusion
            )
            if self.bbox_fusion == "emb_concat":
                self.fc = nn.Linear(output_dim * 5, output_dim)
            elif self.bbox_fusion == "emb_add":
                self.fc = nn.Linear(output_dim * 2, output_dim)

        elif self.lb_fusion == "add":
            assert self.bbox_fusion == "emb_add"
            self.label_emb = nn.Embedding(num_classes, output_dim)
            self.bbox_emb = BboxEncoder(
                num_classes, output_dim, fusion=self.bbox_fusion
            )
        else:
            raise NotImplementedError

    def forward(self, inputs: Dict[str, LongTensor]) -> FloatTensor:
        h_label = self.label_emb(inputs["label"])
        h_bbox = self.bbox_emb(inputs["bbox"])
        if self.lb_fusion == "concat_fc":
            h = torch.cat([h_label, h_bbox], dim=-1)
            h = self.fc(h)
        elif self.lb_fusion == "add":
            h = h_label + h_bbox
        else:
            raise NotImplementedError
        if "mask" in inputs:
            mask_float = inputs["mask"].float()
            mask_float = rearrange(mask_float, "b s -> b s 1")
            h *= mask_float
        return h


class LayoutDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.linear_label = nn.Linear(input_dim, num_classes, bias=False)
        self.linear_bbox = nn.Linear(input_dim, 4 * num_classes, bias=False)

    def forward(self, h: FloatTensor) -> Dict[str, FloatTensor]:
        outputs = {}
        outputs["logit_label"] = self.linear_label(h)  # (B, S, C)
        logit_bbox = self.linear_bbox(h)
        outputs["logit_bbox"] = rearrange(logit_bbox, "b s (c x) -> b s c x", x=4)
        return outputs
