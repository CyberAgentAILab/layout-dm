import logging
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import LongTensor, Tensor
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import sample
from trainer.helpers.task import (
    duplicate_cond,
    set_additional_conditions_for_refinement,
)
from trainer.models.common.nn_lib import (
    CategoricalAggregatedTransformer,
    CategoricalTransformer,
)
from trainer.models.common.util import get_dim_model, shrink

from .logit_adjustment import update
from .util import LOG_EPS, alpha_schedule, index_to_log_onehot, log_onehot_to_index

logger = logging.getLogger(__name__)


class BaseMaskAndReplaceDiffusion(torch.nn.Module):
    """
    Reference: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/4d4cbefe3ed917ec2953af5879aa7608a171b91f/labml_nn/diffusion/ddpm
    Notation is strictly following DDPM paper to avoid confusion
    """

    def __init__(
        self,
        backbone_cfg: DictConfig,
        num_classes: int,
        max_token_length: int,
        num_timesteps: int = 100,
        transformer_type: str = "flattened",
        pos_emb: str = "elem_attr",
        auxiliary_loss_weight: float = 1e-1,
        att_1: float = 0.99999,
        att_T: float = 0.000009,
        ctt_1: float = 0.000009,
        ctt_T: float = 0.99999,
        tokenizer: LayoutSequenceTokenizer = None,
        train_sampling: str = "gumbel",
    ) -> None:
        super().__init__()
        assert transformer_type in ["flattened", "aggregated"]
        assert train_sampling in ["random", "gumbel"]

        self.num_classes = num_classes
        self.max_token_length = max_token_length
        self.num_timesteps = num_timesteps
        self.transformer_type = transformer_type
        self.tokenizer = tokenizer
        self.train_sampling = train_sampling

        self.alpha_schedule_partial_func = partial(
            alpha_schedule,
            num_timesteps=num_timesteps,
            att_1=att_1,
            att_T=att_T,
            ctt_1=ctt_1,
            ctt_T=ctt_T,
        )

        kwargs = {}
        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = len(tokenizer.var_names)

        if "flattened" in transformer_type:
            backbone = instantiate(backbone_cfg)
            self.transformer = CategoricalTransformer(
                backbone=backbone,
                num_classes=self.num_classes,
                max_token_length=max_token_length,
                dim_model=get_dim_model(backbone_cfg),
                pos_emb=pos_emb,
                **kwargs,
            )
        elif transformer_type == "aggregated":
            backbone_cfg = shrink(backbone_cfg, 27 / 29)
            backbone = instantiate(backbone_cfg)
            self.transformer = CategoricalAggregatedTransformer(
                backbone=backbone,
                num_classes=self.num_classes,
                max_len=max_token_length,
                dim_model=get_dim_model(backbone_cfg),
            )

        self.alpha_init_type = "alpha1"
        self.loss_type = "vb_stochastic"
        self.parametrization = "x0"
        self.mask_weight = [1.0, 1.0]
        self.adaptive_auxiliary_loss = True

        self.num_timesteps = num_timesteps
        self.auxiliary_loss_weight = auxiliary_loss_weight

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))
        self.zero_vector = None

    @property
    def device(self) -> torch.device:
        if hasattr(self, "transformer"):
            return next(self.transformer.parameters()).device
        else:
            raise NotImplementedError

    def multinomial_kl(self, log_prob1, log_prob2):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self):  # q(xt|xt_1)
        raise NotImplementedError

    def q_pred(self):  # q(xt|x0)
        raise NotImplementedError

    def predict_start(self, log_x_t, t):  # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        out = self.transformer(x_t, timestep=t)["logits"]

        out = out[:, :, :-1]  # ignore MASK
        out = rearrange(out, "b s c -> b c s")

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes - 1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = (
                torch.zeros(batch_size, 1, self.max_token_length).type_as(log_x_t) - 70
            )
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(
        self, log_x_start, log_x_t, t
    ):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        raise NotImplementedError

    @torch.no_grad()
    def p_sample(
        self, log_x: Tensor, t: Tensor, sampling_cfg: Optional[DictConfig] = None
    ):
        # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, t)

        # for compatibility with other approaches
        out_index = sample(model_log_prob, sampling_cfg)
        out_index = rearrange(out_index, "b 1 s -> b s")
        out = index_to_log_onehot(out_index, self.num_classes)

        return out

    def log_sample_categorical(
        self, logits: Tensor
    ):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self):  # diffusion step, q(xt|x0) and sample xt
        raise NotImplementedError

    def sample_time(self, b: int, device: torch.device, method: str = "uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def forward(self, x, is_train=True):  # get the KL loss
        raise NotImplementedError

    def _sample_single_step(
        self,
        log_z: Tensor,
        model_t: Tensor,
        skip_step: int,
        sampling_cfg: Optional[DictConfig] = None,
        cond: Optional[Dict] = None,
    ):
        with torch.no_grad():
            # Infer x0 at first
            log_x_recon = self.predict_start(log_z, model_t)

            # add less noise!
            time_difference = getattr(sampling_cfg, "time_difference", 0.0)
            if time_difference > 0.0:
                T = self.num_timesteps
                noise_t = torch.clamp(
                    model_t - int(T * time_difference), 0, T - 1
                ).long()
            else:
                noise_t = model_t.clone()

            if skip_step > 0:
                if noise_t[0].item() > skip_step:
                    model_log_prob = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z, t=noise_t - skip_step
                    )
                else:
                    model_log_prob = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z, t=noise_t
                    )
            else:
                # model_log_prob = self.p_pred(log_z, t)
                model_log_prob = self.q_posterior(
                    log_x_start=log_x_recon, log_x_t=log_z, t=noise_t
                )

        # adjust logits distribution based on some gradient in logit space
        if cond:
            # impose strong user-specified constraints by replacement
            if "mask" in cond:
                with torch.no_grad():
                    strong_mask = rearrange(cond["mask"], "b s -> b 1 s")
                    strong_log_prob = index_to_log_onehot(cond["seq"], self.num_classes)
                    model_log_prob = torch.where(
                        strong_mask, strong_log_prob, model_log_prob
                    )

            # logit adjustment by hand-crafted rules
            if cond.get("type", None) == "refinement":
                with torch.no_grad():
                    model_log_prob[cond["weak_mask"]] += cond["weak_logits"][
                        cond["weak_mask"]
                    ]

            # logit adjustment by gradients from loss functions
            if cond.get("type", None) == "relation":
                t = model_t[0].item()
                model_log_prob = update(
                    t=t,
                    cond=cond,
                    model_log_prob=model_log_prob,
                    tokenizer=self.tokenizer,
                    sampling_cfg=sampling_cfg,
                )

            # disable [PAD] when the number of elements is known
            if cond["type"] in ["c", "cwh", "refinement", "relation"]:
                with torch.no_grad():
                    step = self.tokenizer.N_var_per_element
                    B, S = cond["seq"].size()
                    pad_id = self.tokenizer.name_to_id("pad")
                    attr_indices = repeat(torch.arange(S), "s -> b s", b=B).to(
                        model_log_prob.device
                    )
                    pad_mask = (attr_indices % step != 0) & (cond["seq"] != pad_id)
                    pad_mask = repeat(pad_mask, "b s -> b c s", c=self.num_classes)
                    index = rearrange(torch.arange(self.num_classes), "c -> 1 c 1")
                    pad_mask = pad_mask & (index.to(self.device) == pad_id)
                    model_log_prob[pad_mask] = LOG_EPS

        with torch.no_grad():
            out_index = sample(model_log_prob, sampling_cfg)
            out_index = rearrange(out_index, "b 1 s -> b s")
            log_z = index_to_log_onehot(out_index, self.num_classes)

        return log_z

    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        get_intermediate_results: bool = False,
        **kwargs,
    ) -> Union[LongTensor, List[LongTensor]]:
        """
        cond["mask"] is for forcing the model to use user-provided inputs in each step
        """

        if cond and cond["type"] == "refinement":
            cond = set_additional_conditions_for_refinement(
                cond, self.tokenizer, sampling_cfg
            )

        num_timesteps_eval = sampling_cfg.get("num_timesteps", self.num_timesteps)
        assert num_timesteps_eval <= self.num_timesteps
        diffusion_list = []
        for i in range(num_timesteps_eval - 1, -1, -1):
            diffusion_list.append(int(i * self.num_timesteps / num_timesteps_eval))
        prev_diffusion_index = self.num_timesteps  # set very large value

        device = self.device
        if get_intermediate_results:
            results = []

        if cond:
            cond = duplicate_cond(cond, batch_size)

            # this is used in demo to see the variety
            multiple_outputs_from_single_cond = cond["seq"].size(0) == 1
            multiple_outputs_from_single_cond &= batch_size > 1

            for k in cond:
                if isinstance(cond[k], Tensor):
                    cond[k] = cond[k].to(device)

            if multiple_outputs_from_single_cond:
                sizes = [
                    batch_size,
                ] + [1 for _ in range(cond[k].dim() - 1)]
                cond[k] = cond[k].repeat(sizes)
            log_z = index_to_log_onehot(cond["seq"], self.num_classes)
        else:
            zero_logits = torch.zeros(
                (batch_size, self.num_classes - 1, self.max_token_length), device=device
            )
            one_logits = torch.ones(
                (batch_size, 1, self.max_token_length), device=device
            )
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z = torch.log(mask_logits)

        for diffusion_index in diffusion_list:
            delta_t = prev_diffusion_index - diffusion_index
            if delta_t > 0:
                t = torch.full(
                    (batch_size,), diffusion_index, device=device, dtype=torch.long
                )
                log_z = self._sample_single_step(
                    log_z=log_z,
                    model_t=t,
                    skip_step=delta_t - 1,
                    sampling_cfg=sampling_cfg,
                    cond=cond,  # used to inject use-specified inputs
                )
            else:
                raise NotImplementedError

            if get_intermediate_results:
                results.append(log_z.cpu())
            prev_diffusion_index = diffusion_index

        if get_intermediate_results:
            return [log_onehot_to_index(r) for r in results]
        else:
            return log_onehot_to_index(log_z).cpu()
