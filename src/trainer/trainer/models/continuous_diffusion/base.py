import logging
import math
import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch import FloatTensor, LongTensor, Tensor
from torch.special import expm1
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.models.common.nn_lib import ContinuousTransformer
from trainer.models.transformer_utils import TransformerEncoder

logger = logging.getLogger(__name__)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log(t: FloatTensor, eps=1e-20) -> FloatTensor:
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x: Tensor, t: Tensor):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t: FloatTensor) -> FloatTensor:
    return -torch.log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t: FloatTensor, s: float = 0.008) -> FloatTensor:
    return -log(
        (torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5
    )  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr: FloatTensor) -> FloatTensor:
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


def _query(
    key: str,
    any_cfg: DictConfig,
    default_class: Optional[Any] = None,
    default_value: Optional[Any] = None,
):
    if key in any_cfg:
        return getattr(any_cfg, key)
    else:
        if default_class is not None:
            return getattr(default_class, key)
        elif default_value is not None:
            return default_value
        else:
            raise NotImplementedError


def init_token_embedding(
    num_embeddings: int,
    embedding_dim: int,
    is_learnable: bool = False,
    # std_scale: float = 0.5,
    std_scale: float = 1.0,
) -> torch.nn.Module:
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    # data manifold roughly in [-1.0, 1.0]
    nn.init.trunc_normal_(embedding.weight.data, 0.0, std_scale)
    if not is_learnable:
        for param in embedding.parameters():
            param.requires_grad = False
    return embedding


class ContinuousDiffusionBase(torch.nn.Module):
    def __init__(
        self,
        backbone: TransformerEncoder,
        max_len: int,
        dim_model: int,
        pos_emb: str = "default",
        num_timesteps: int = 100,
        use_self_condition: bool = False,
        noise_schedule: str = "cosine",
        time_difference: float = 0.0,
        tokenizer: LayoutSequenceTokenizer = None,
        num_channel: int = 16,
        learnable_token_emb: bool = False,
        use_clamping_trick: bool = False,
        use_token_emb_normalization: bool = False,
    ) -> None:
        """
        This implementation largely rely on https://github.com/lucidrains/bit-diffusion
        """
        super().__init__()

        self.max_len = max_len
        self.pos_emb = pos_emb
        self.num_timesteps = num_timesteps
        self.use_self_condition = use_self_condition
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")
        self.time_difference = time_difference
        self.tokenizer = tokenizer
        self.num_channel = num_channel
        self.learnable_token_emb = learnable_token_emb
        self.use_clamping_trick = use_clamping_trick
        self.use_token_emb_normalization = use_token_emb_normalization
        self.scale = None

        kwargs = {}
        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = tokenizer.N_var_per_element

        self.transformer = ContinuousTransformer(
            backbone=backbone,
            dim_model=dim_model,
            dim_in=num_channel,
            max_token_length=max_len,
            lookahead=True,
            pos_emb=pos_emb,
            **kwargs,
        )

        # common across models unlike rounder
        self.token_emb = init_token_embedding(
            num_embeddings=tokenizer.N_total,
            embedding_dim=num_channel,
            is_learnable=learnable_token_emb,
        )

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def forward(
        self, inputs: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        # make sure everything is on GPU
        B = inputs["seq"].size(0)
        seq = inputs["seq"].to(self.device)

        x, x_mean = self.dis2con(
            seq, reparametrize=True, normalize=self.use_token_emb_normalization
        )
        times = torch.zeros((B,), device=self.device).float().uniform_(0, 0.999)
        noise = torch.randn_like(x)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(x, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        noised_x = alpha * x + sigma * noise

        self_cond = None
        if self.use_self_condition and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.transformer(x=noised_x, timestep=noise_level)
                self_cond = self_cond["outputs"].detach_()

        outputs = self.transformer(
            x=noised_x, timestep=noise_level, x_self_cond=self_cond
        )

        # trying to accurately replicate DiffusionLM's loss (consisting of three terms)
        # https://github.com/XiangLi1999/Diffusion-LM/blob/352b28c623c83083ab31d0d0b2a611f01c0fcb6c/improved-diffusion/improved_diffusion/gaussian_diffusion.py#L1557-L1579
        mse_loss = F.mse_loss(outputs["outputs"], x, reduction="none")

        t0_loss = F.mse_loss(outputs["outputs"], x_mean, reduction="none")
        t0_mask = times < 1 / self.num_timesteps
        mse_loss = torch.where(rearrange(t0_mask, "b -> b 1 1"), t0_loss, mse_loss)

        final_times = torch.full((1,), fill_value=0.999, device=self.device).float()
        final_noise_level = self.log_snr(final_times)
        final_padded_noise_level = right_pad_dims_to(x, final_noise_level)
        final_alpha, _ = log_snr_to_alpha_sigma(final_padded_noise_level)
        tT_x = final_alpha * x
        tT_loss = tT_x**2

        losses = {
            "mse_loss": mse_loss.mean(),
            "tT_loss": tT_loss.mean(),  # avoid w being too large (trivial solution)
        }

        if hasattr(self, "con2logits") and self.con2logits is not None:
            logits = self.con2logits(outputs["outputs"])
            logits = rearrange(logits, "b s c -> b c s")
            losses["rounding_loss"] = F.cross_entropy(logits, seq)

        return outputs, losses

    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        get_intermediate_results: bool = False,
        **kwargs,
    ) -> Union[FloatTensor, List[FloatTensor]]:
        if cond:
            for k, v in cond.items():
                if isinstance(v, Tensor):
                    cond[k] = v.to(self.device)
            cond["arr"] = self.dis2con(
                cond["seq"], normalize=self.use_token_emb_normalization
            )
            cond["mask"] = repeat(cond["mask"], "b s -> b s c", c=cond["arr"].size(-1))

        if "use_ddim" in sampling_cfg and sampling_cfg["use_ddim"]:
            arr = self.sample_ddim(
                batch_size=batch_size,
                cond=cond,
                sampling_cfg=sampling_cfg,
                get_intermediate_results=get_intermediate_results,
                **kwargs,
            )
        else:
            arr = self.sample_ddpm(
                batch_size=batch_size,
                cond=cond,
                sampling_cfg=sampling_cfg,
                get_intermediate_results=get_intermediate_results,
                **kwargs,
            )
        seq = self.con2dis(arr)
        return seq

    def get_sampling_timesteps(
        self, batch: int, num_timesteps: int, *, device: torch.device
    ):
        times = torch.linspace(1.0, 0.0, num_timesteps + 1, device=device)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        get_intermediate_results: bool = False,
        **kwargs,
    ) -> Union[FloatTensor, List[FloatTensor]]:
        num_inference_steps = _query("num_timesteps", sampling_cfg, default_value=100)

        time_pairs = self.get_sampling_timesteps(
            batch=batch_size, num_timesteps=num_inference_steps, device=self.device
        )

        x = torch.randn(
            (batch_size, self.max_len, self.num_channel), device=self.device
        )
        if cond:
            x = torch.where(cond["mask"], cond["arr"], x)
        x_start = None
        td = _query("time_difference", sampling_cfg, default_class=self)

        for times, times_next in time_pairs:
            # add the time delay
            times_next = (times_next - td).clamp(min=0.0)

            # get times and noise levels
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(
                partial(right_pad_dims_to, x), (log_snr, log_snr_next)
            )

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # predict x0
            if not self.use_self_condition:
                x_start = None
            x_start = self.transformer(x=x, timestep=log_snr, x_self_cond=x_start)[
                "outputs"
            ]
            x_start = (
                self.clamping_trick(x_start) if self.use_clamping_trick else x_start
            )

            # get predicted noise
            pred_noise = (x - alpha * x_start) / sigma.clamp(min=1e-8)

            # calculate x next
            x = x_start * alpha_next + pred_noise * sigma_next

            # use conditiong
            if cond:
                x = torch.where(cond["mask"], cond["arr"], x)

        return x

    @torch.no_grad()
    def sample_ddpm(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        get_intermediate_results: bool = False,
        **kwargs,
    ) -> Union[FloatTensor, List[FloatTensor]]:
        num_inference_steps = _query("num_timesteps", sampling_cfg, default_value=100)

        time_pairs = self.get_sampling_timesteps(
            batch=batch_size, num_timesteps=num_inference_steps, device=self.device
        )

        x = torch.randn(
            (batch_size, self.max_len, self.num_channel), device=self.device
        )
        if cond:
            x = torch.where(cond["mask"], cond["arr"], x)
        x_start = None
        td = _query("time_difference", sampling_cfg, default_class=self)

        for time, time_next in time_pairs:
            time_next = (time_next - td).clamp(min=0.0)
            noise_cond = self.log_snr(time)

            # get predicted x0
            if not self.use_self_condition:
                x_start = None
            x_start = self.transformer(x=x, timestep=noise_cond, x_self_cond=x_start)[
                "outputs"
            ]
            x_start = (
                self.clamping_trick(x_start) if self.use_clamping_trick else x_start
            )

            # get log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, x), (log_snr, log_snr_next)
            )

            # get alpha sigma of time and next time
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
            variance = (sigma_next**2) * c
            log_variance = log(variance)

            # get noise
            noise = torch.where(
                rearrange(time_next > 0, "b -> b 1 1"),
                torch.randn_like(x),
                torch.zeros_like(x),
            )
            x = mean + (0.5 * log_variance).exp() * noise

            if cond:
                x = torch.where(cond["mask"], cond["arr"], x)

        return x

    def dis2con(
        self, seq: LongTensor, reparametrize: bool = False, normalize: bool = False
    ) -> Union[FloatTensor, Tuple[FloatTensor, FloatTensor]]:
        raise NotImplementedError

    def con2dis(self, arr: FloatTensor) -> LongTensor:
        raise NotImplementedError

    def con2logits(self, arr: FloatTensor) -> LongTensor:
        raise NotImplementedError

    def clamping_trick(self, x_start: FloatTensor, reparametrize=False):
        # clamping trick to reduce rounding errors
        logits = self.con2logits(x_start)
        clamped_seq = torch.argmax(logits, dim=-1)
        if reparametrize:
            x_start, _ = self.dis2con(
                clamped_seq,
                reparametrize=True,
                normalize=self.use_token_emb_normalization,
            )
        else:
            x_start = self.dis2con(
                clamped_seq,
                reparametrize=False,
                normalize=self.use_token_emb_normalization,
            )

        if self.scale:
            x_start.clamp_(-self.scale, self.scale)
        return x_start
