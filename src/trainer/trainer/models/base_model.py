import logging
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from trainer.helpers.layout_tokenizer import LayoutTokenizer

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    # def sample(self, z: Optional[Tensor]):
    def sample(self):
        """
        Generate sample based on z.
        z can be either given or sampled from some distribution (e.g., normal)
        """
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        if hasattr(self, "model"):
            return next(self.model.parameters()).device
        else:
            raise NotImplementedError

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: LayoutTokenizer):
        self._tokenizer = value

    def compute_stats(self):
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def optim_groups(
        self, weight_decay: float = 0.0, additional_no_decay: Optional[List[str]] = None
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        # see https://github.com/kampta/DeepLayout/blob/main/layout_transformer/model.py#L139
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.modules.activation.MultiheadAttention,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if additional_no_decay:
            for k in additional_no_decay:
                no_decay.add(k)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def update_per_epoch(self, epoch: int, max_epoch: int):
        """
        Update some non-trainable parameters during training (e.g., warmup)
        """
        pass

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: DictConfig
    ) -> DictConfig:
        """
        Set user-specified args for sampling cfg
        """
        # Aggregate refinement-related parameters
        is_ruite = type(self).__name__ == "RUITE"
        if args.cond == "refinement" and args.refine_lambda > 0.0 and not is_ruite:
            sampling_cfg.refine_mode = args.refine_mode
            sampling_cfg.refine_offset_ratio = args.refine_offset_ratio
            sampling_cfg.refine_lambda = args.refine_lambda

        if args.cond == "relation" and args.relation_lambda > 0.0:
            sampling_cfg.relation_mode = args.relation_mode
            sampling_cfg.relation_lambda = args.relation_lambda
            sampling_cfg.relation_tau = args.relation_tau
            sampling_cfg.relation_num_update = args.relation_num_update

        if "num_timesteps" not in sampling_cfg:
            # for dec or enc-dec
            if "eos" in self.tokenizer.special_tokens:
                sampling_cfg.num_timesteps = self.tokenizer.max_token_length
            else:
                sampling_cfg.num_timesteps = args.num_timesteps

        return sampling_cfg
