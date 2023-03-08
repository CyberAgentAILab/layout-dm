import copy
import logging
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import sample
from trainer.helpers.task import duplicate_cond, get_cond
from trainer.helpers.util import batch_shuffle_index
from trainer.models.base_model import BaseModel
from trainer.models.common.nn_lib import (
    CategoricalEncDecTransformer,
    CustomDataParallel,
)
from trainer.models.common.util import get_dim_model, shrink

logger = logging.getLogger(__name__)


class BART(BaseModel):
    """
    To reproduce
    BLT: Bidirectional Layout Transformer for Controllable Layout Generation (ECCV2022)
    https://arxiv.org/abs/2112.05112
    """

    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        sort_by: str = "none",  # "category_alphabetical"
        tasks: Union[str, List[str]] = [
            "random",
        ],  # ["c", "cwh", "partial"],
        pos_emb: str = "default",
    ) -> None:
        super().__init__()
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = tasks

        kwargs = {}
        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = tokenizer.N_var_per_element

        self.tokenizer = tokenizer
        self.tokenizer.sort_by = sort_by
        assert self.tokenizer.var_order == "c-w-h-x-y"
        assert self.tokenizer.special_tokens == ["pad", "bos", "eos", "mask"]

        # Note: make sure learnable parameters are inside self.model
        # backbone = instantiate(backbone_cfg)
        # dim_model=get_dim_model(backbone_cfg)

        backbone_enc_cfg = shrink(backbone_cfg, 21 / 32)
        backbone_dec_cfg = shrink(backbone_cfg, 21 / 32)
        backbone_enc = instantiate(backbone_enc_cfg)

        params = {
            k: v
            for (k, v) in backbone_dec_cfg["encoder_layer"].items()
            if k != "_target_"
        }
        backbone_dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(**params),
            num_layers=backbone_dec_cfg["num_layers"],
        )

        self.model = CustomDataParallel(
            CategoricalEncDecTransformer(
                backbone_enc=backbone_enc,
                backbone_dec=backbone_dec,
                dim_model=get_dim_model(backbone_enc_cfg),
                num_classes_dec=self.tokenizer.N_total,
                max_token_length_dec=tokenizer.max_token_length + 1,
                pos_emb=pos_emb,
                **kwargs,
            )
        )
        self.apply(self._init_weights)
        self.compute_stats()
        self.loss_fn_ce = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.name_to_id("pad")
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        outputs = self.model(input=inputs["input"], target=inputs["target"][:, :-1])
        nll_loss = self.loss_fn_ce(
            rearrange(outputs["logits"], "b s c -> b c s"),
            inputs["target"][:, 1:],
        )
        losses = {"nll_loss": nll_loss}
        outputs["ids"] = torch.argmax(outputs["logits"], dim=-1)
        return outputs, losses

    def sample(
        self,
        batch_size: Optional[int],
        cond: Optional[Tensor] = None,
        sampling_cfg: Optional[DictConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Generate sample based on z.
        z can be either given or sampled from some distribution (e.g., normal)
        """
        if "cond_type" in kwargs:
            cond_type = kwargs["cond_type"]
        else:
            cond_type = random.choice(self.tasks)

        is_special = cond_type == "partial"
        is_special = is_special and (self.tokenizer.sort_by == "category_alphabetical")

        bos_id = self.tokenizer.name_to_id("bos")
        mask_id = self.tokenizer.name_to_id("mask")

        B, S = (batch_size, self.tokenizer.max_token_length)
        C = self.tokenizer.N_var_per_element
        dec_input = torch.full((batch_size, 1), fill_value=bos_id)

        if cond:
            # **_user will not be updated (kept as reference)
            cond = duplicate_cond(cond, batch_size)
            enc_input = cond["seq"].clone()

            if is_special:
                partial_inputs = []
                for b in range(B):
                    ids = cond["seq"][b][cond["mask"][b]][1:]
                    ids = rearrange(ids, "(x c) -> x c", x=len(ids) // C, c=C)
                    partial_inputs.append(ids)
                partial_inputs_copy = copy.deepcopy(partial_inputs)
            else:
                seq_user = cond["seq"].clone()
                mask_user = cond["mask"].clone()
        else:
            enc_input = torch.full((batch_size, 1), fill_value=bos_id)
            enc_input = torch.cat(
                [
                    enc_input,
                    torch.full((B, S), fill_value=mask_id),
                ],
                dim=1,
            )

        for i in range(S):
            logits = self.model(enc_input.to(device), dec_input.to(device))[
                "logits"
            ].cpu()
            logits = rearrange(logits[:, i : i + 1], "b 1 c -> b c")

            # constrained decoding
            invalid = repeat(~self.tokenizer.token_mask[i : i + 1], "1 c -> b c", b=B)
            if (
                self.tokenizer.sort_by == "category_alphabetical"
                and i // C > 0
                and i % C == 0
            ):
                # limit the logits to follow the alphabetical sorting order
                indices = repeat(torch.arange(self.tokenizer.N_total), "c -> b c", b=B)
                invalid = invalid | (indices < dec_input[:, i - 4 : i - 3])
            logits[invalid] = -float("Inf")

            predicted = sample(logits, sampling_cfg)

            if cond and not is_special:
                # note: [:, 0] is [BOS]
                id_ = seq_user[:, i + 1 : (i + 1) + 1]
                flag = ~mask_user[:, i + 1 : (i + 1) + 1]
                if id_.size(1) == 1:
                    # If condition exists and is valid, use it
                    predicted = torch.where(flag, predicted, id_)
            dec_input = torch.cat([dec_input, predicted], dim=1)

            if cond and is_special:
                if (i + 1) % C != 0:
                    continue
                for b in range(B):
                    if partial_inputs[b].size(0) == 0:
                        continue
                    category = partial_inputs[b][0, 0]
                    start, stop = i + 2 - C, i + 2
                    if dec_input[b, start].item() >= category.item():
                        dec_input[b, start:stop] = partial_inputs[b][0]
                        partial_inputs[b] = partial_inputs[b][1:]  # pop copied element

        if cond and is_special:
            for b, p in enumerate(partial_inputs):
                if p.size(0) == 0:
                    continue
                while True:
                    if partial_inputs[b].size(0) == 0:
                        break
                    # randomly pick an element for replacement
                    ind = random.randint(0, self.tokenizer.max_seq_length - 1)
                    start, stop = ind * C + 1, (ind + 1) * C + 1
                    tgt = dec_input[b][start:stop]

                    # if the sampled element is one of the partial input, retry sampling
                    retry = any((e == tgt).all().item() for e in partial_inputs_copy[b])
                    if not retry:
                        dec_input[b][start:stop] = partial_inputs[b][0]
                        partial_inputs[b] = partial_inputs[b][1:]  # pop copied element

        seq = dec_input[:, 1:].cpu()  # pop BOS
        layouts = self.tokenizer.decode(seq)
        return layouts

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)

        data = self.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        # input should be alphabetically ordered (note: pos should be randomly shuffled)
        # note: do not consider padding_mask since it may leak info. (e.g., num of elements)
        task = random.choice(self.tasks)

        if task == "unconditional":  # keep BOS only
            input = copy.deepcopy(data["seq"])
            input[:, 1:] = self.tokenizer.name_to_id("mask")
        else:
            input = get_cond(
                batch=batch,
                tokenizer=self.tokenizer,
                cond_type=task,
                model_type=type(self).__name__,
            )["seq"]
            if (
                self.tasks == ["random"]
                and self.tokenizer.sort_by == "category_alphabetical"
            ):
                # randomly shuffle the inputs (except BOS) to avoid leaking number of elements of each category in documents
                B, S = data["seq"][:, 1:].size()
                C = self.tokenizer.N_var_per_element
                feature_length = S // C
                index = batch_shuffle_index(batch_size=B, feature_length=feature_length)
                index = repeat(index, "b s -> b (s c)", c=C) * C
                index += repeat(torch.arange(C), "c -> b (f c)", b=B, f=feature_length)
                data["seq"][:, 1:] = torch.gather(input=data["seq"], dim=1, index=index)

        return {
            "target": data["seq"],
            "input": input,
        }

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        additional_no_decay = []
        for base in ["input_pos_emb", "target_pos_emb"]:
            for n in getattr(self.model.module, base).no_decay_param_names:
                additional_no_decay.append(f"model.module.{base}.{n}")
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
