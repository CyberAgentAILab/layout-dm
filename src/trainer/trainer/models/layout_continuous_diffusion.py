import logging
from typing import Dict, Iterable, Optional, Tuple, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.models.base_model import BaseModel
from trainer.models.common.nn_lib import CustomDataParallel
from trainer.models.common.util import get_dim_model, shrink
from trainer.models.continuous_diffusion.base import init_token_embedding
from trainer.models.continuous_diffusion.bitdiffusion import BitDiffusion
from trainer.models.continuous_diffusion.diffusion_lm import DiffusionLM

logger = logging.getLogger(__name__)

BITS = 8
MODELS = {"bit_diffusion": BitDiffusion, "diffusion_lm": DiffusionLM}


class LayoutContinuousDiffusion(BaseModel):
    def __init__(
        self,
        # cfg: DictConfig,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        model_type: str,
        num_channel: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_channel = num_channel

        self.max_len = tokenizer.max_token_length

        # make sure MASK is the last vocabulary
        assert tokenizer.id_to_name(tokenizer.N_total - 1) == "mask"
        self.tokenizer = tokenizer

        model = MODELS[model_type]
        # Note: make sure learnable parameters are inside self.model
        backbone_shrink_cfg = shrink(backbone_cfg, 29 / 32)
        backbone = instantiate(backbone_shrink_cfg)  # for fair comparison
        self.model = CustomDataParallel(
            model(
                backbone=backbone,
                tokenizer=tokenizer,
                dim_model=get_dim_model(backbone_shrink_cfg),
                max_len=self.max_len,
                num_channel=self.num_channel,
                **kwargs,
            )
        )
        self.apply(self._init_weights)

        if model_type == "diffusion_lm":
            # re-initialize to avoid weight range specification
            self.model.module.token_emb = init_token_embedding(
                num_embeddings=tokenizer.N_total,
                embedding_dim=num_channel,
                is_learnable=self.model.module.learnable_token_emb,
            )
            # initialize rounder by an inverse of token_emb
            self.model.module.rounder.weight.data = (
                self.model.module.token_emb.weight.data.clone()
            )

        self.compute_stats()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        outputs, losses = self.model(inputs)
        # aggregate losses for multi-GPU mode (no change in single GPU mode)
        new_losses = {k: v.mean() for (k, v) in losses.items()}
        return outputs, new_losses

    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        seq = self.model.sample(
            batch_size=batch_size, cond=cond, sampling_cfg=sampling_cfg, **kwargs
        ).cpu()
        return self.tokenizer.decode(seq)

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: DictConfig
    ) -> DictConfig:
        sampling_cfg = super().aggregate_sampling_settings(sampling_cfg, args)

        sampling_cfg.use_ddim = args.use_ddim
        sampling_cfg.time_difference = args.time_difference

        return sampling_cfg

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        inputs = {"label": label, "mask": mask, "bbox": bbox}
        return self.tokenizer.encode(inputs)

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        base = "model.module.transformer.pos_emb"
        additional_no_decay = [
            f"{base}.{name}"
            for name in self.model.module.transformer.pos_emb.no_decay_param_names
        ]
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
