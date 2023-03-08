import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import Tensor  # for type hints
from trainer.helpers.layout_tokenizer import Converter, LayoutSequenceTokenizer
from trainer.helpers.sampling import RandomSamplingConfig, sample

from .base import BaseMaskAndReplaceDiffusion
from .util import (
    extract,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_categorical,
    log_onehot_to_index,
    mean_except_batch,
)

logger = logging.getLogger(__name__)


class ConstrainedMaskAndReplaceDiffusion(BaseMaskAndReplaceDiffusion):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        num_classes: int,
        max_token_length: int,
        num_timesteps: int = 100,
        tokenizer: LayoutSequenceTokenizer = None,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_cfg=backbone_cfg,
            num_classes=num_classes,
            max_token_length=max_token_length,
            num_timesteps=num_timesteps,
            tokenizer=tokenizer,
            **kwargs,
        )

        # tokenizer is required to separate corruption matrix
        self.tokenizer = tokenizer
        self.converter = Converter(self.tokenizer)

        # set vocabulari size for each corruption matrix (w/ pad)
        self.mat_size = {"c": self.tokenizer.N_category + 2}
        num_bin = self.tokenizer.N_bbox_per_var
        for key in ["x", "y", "w", "h"]:
            self.mat_size.update({key: num_bin + 2})

        for key in self.tokenizer.var_names:
            if self.alpha_init_type == "alpha1":
                N = self.mat_size[key] - 1
                at, bt, ct, att, btt, ctt = self.alpha_schedule_partial_func(N=N)
            else:
                print("alpha_init_type is Wrong !! ")
                raise NotImplementedError

            log_at, log_bt, log_ct = torch.log(at), torch.log(bt), torch.log(ct)
            log_cumprod_at, log_cumprod_bt, log_cumprod_ct = (
                torch.log(att),
                torch.log(btt),
                torch.log(ctt),
            )

            log_1_min_ct = log_1_min_a(log_ct)
            log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

            assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
            assert (
                log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item()
                < 1.0e-5
            )

            # Convert to float32 and register buffers.
            self.register_buffer(f"{key}_log_at", log_at.float())
            self.register_buffer(f"{key}_log_bt", log_bt.float())
            self.register_buffer(f"{key}_log_ct", log_ct.float())
            self.register_buffer(f"{key}_log_cumprod_at", log_cumprod_at.float())
            self.register_buffer(f"{key}_log_cumprod_bt", log_cumprod_bt.float())
            self.register_buffer(f"{key}_log_cumprod_ct", log_cumprod_ct.float())
            self.register_buffer(f"{key}_log_1_min_ct", log_1_min_ct.float())
            self.register_buffer(
                f"{key}_log_1_min_cumprod_ct", log_1_min_cumprod_ct.float()
            )

    def q_pred_one_timestep(
        self, log_x_t: Tensor, t: Tensor, key: str
    ) -> Tensor:  # q(xt|xt_1)
        log_at = extract(getattr(self, f"{key}_log_at"), t, log_x_t.shape)  # at
        log_bt = extract(getattr(self, f"{key}_log_bt"), t, log_x_t.shape)  # bt
        log_ct = extract(getattr(self, f"{key}_log_ct"), t, log_x_t.shape)  # ct
        log_1_min_ct = extract(
            getattr(self, f"{key}_log_1_min_ct"), t, log_x_t.shape
        )  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: Tensor, key: str) -> Tensor:  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        s = log_x_start.shape
        log_cumprod_at = extract(getattr(self, f"{key}_log_cumprod_at"), t, s)  # at~
        log_cumprod_bt = extract(getattr(self, f"{key}_log_cumprod_bt"), t, s)  # bt~
        log_cumprod_ct = extract(getattr(self, f"{key}_log_cumprod_ct"), t, s)  # ct~
        log_1_min_cumprod_ct = extract(
            getattr(self, f"{key}_log_1_min_cumprod_ct"), t, s
        )  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),
            ],
            dim=1,
        )

        return log_probs

    def q_posterior(
        self, log_x_start: Tensor, log_x_t: Tensor, t: Tensor
    ) -> Tensor:  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(self.device)

        log_x_start_full, log_x_t_full = log_x_start, log_x_t  # for API compatibiliry

        batch_size = log_x_start_full.size()[0]
        step = self.tokenizer.N_var_per_element

        index_x_t_full = log_onehot_to_index(log_x_t_full)
        log_one_vector_full = torch.zeros(batch_size, 1, 1).type_as(log_x_t_full)
        seq_len = self.max_token_length // step
        log_zero_vector_full = torch.log(log_one_vector_full + 1.0e-30).expand(
            -1, -1, seq_len
        )
        mask_reshaped = rearrange(
            index_x_t_full == self.tokenizer.name_to_id("mask"),
            "b (s x) -> b s x",
            s=seq_len,
            x=step,
        )

        log_EV_xtmin_given_xt_given_xstart_full = []
        for i, key in enumerate(self.tokenizer.var_names):
            mask = mask_reshaped[..., i].unsqueeze(1)
            log_x_start = self.converter.f_to_p_log(log_x_start_full[..., i::step], key)
            log_x_t = self.converter.f_to_p_log(log_x_t_full[..., i::step], key)
            log_qt = self.q_pred(log_x_t, t, key)  # q(xt|x0)

            log_qt = log_qt[:, :-1, :]
            log_cumprod_ct = extract(
                getattr(self, f"{key}_log_cumprod_ct"), t, log_x_t.shape
            )  # ct~
            ct_cumprod_vector = log_cumprod_ct.expand(-1, self.mat_size[key] - 1, -1)
            log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

            log_qt_one_timestep = self.q_pred_one_timestep(
                log_x_t, t, key
            )  # q(xt|xt_1)

            log_qt_one_timestep = torch.cat(
                (log_qt_one_timestep[:, :-1, :], log_zero_vector_full), dim=1
            )
            log_ct = extract(getattr(self, f"{key}_log_ct"), t, log_x_t.shape)  # ct
            ct_vector = log_ct.expand(-1, self.mat_size[key] - 1, -1)
            ct_vector = torch.cat((ct_vector, log_one_vector_full), dim=1)
            log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

            # below just does log(ab/c)=loga+logb-logc in eq.5 of VQDiffusion
            q = log_x_start[:, :-1, :] - log_qt
            q = torch.cat((q, log_zero_vector_full), dim=1)
            q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
            q = q - q_log_sum_exp
            log_EV_xtmin_given_xt_given_xstart = (
                self.q_pred(q, t - 1, key) + log_qt_one_timestep + q_log_sum_exp
            )
            log_EV_xtmin_given_xt_given_xstart = torch.clamp(
                log_EV_xtmin_given_xt_given_xstart, -70, 0
            )
            log_EV_xtmin_given_xt_given_xstart_full.append(
                self.converter.p_to_f_log(log_EV_xtmin_given_xt_given_xstart, key)
            )

        log_EV_xtmin_given_xt_given_xstart_full = torch.stack(
            log_EV_xtmin_given_xt_given_xstart_full, dim=-1
        ).view(batch_size, self.num_classes, -1)

        return log_EV_xtmin_given_xt_given_xstart_full

    def log_sample_categorical(
        self, logits: Tensor, key: str
    ) -> Tensor:  # use gumbel to sample onehot vector from log probability
        if self.train_sampling == "gumbel":
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sampled = (gumbel_noise + logits).argmax(dim=1)
        elif self.train_sampling == "random":
            sampling_cfg = OmegaConf.structured(RandomSamplingConfig)
            sampled = sample(logits, sampling_cfg)
            sampled = rearrange(sampled, "b 1 s -> b s")

        log_sample = index_to_log_onehot(sampled, self.mat_size[key])
        return log_sample

    def q_sample(
        self, log_x_start: Tensor, t: Tensor, key: str
    ) -> Tensor:  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, key)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0, key)

        return log_sample

    def forward(
        self, x: Tensor, is_train: bool = True
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        b, s = x.size()[:2]
        device = x.device
        step = self.tokenizer.N_var_per_element

        assert self.loss_type == "vb_stochastic"
        x_start_full = x
        t, pt = self.sample_time(b, device, "importance")

        log_x_start_full = index_to_log_onehot(x_start_full, self.num_classes)
        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(self.device)
        x_start_reshaped = self.converter.f_to_p_id_all(
            rearrange(x_start_full, "b (s x) -> b s x", s=s // step, x=step)
        )
        log_x_t_full = []
        xt_full = []
        for i, key in enumerate(self.tokenizer.var_names):
            log_x_start = index_to_log_onehot(
                x_start_reshaped[..., i], self.mat_size[key]
            )
            log_x_t = self.q_sample(log_x_start=log_x_start, t=t, key=key)
            log_x_t_full.append(self.converter.p_to_f_log(log_x_t, key))
            xt_full.append(log_onehot_to_index(log_x_t))

        xt_full = self.converter.p_to_f_id_all(torch.stack(xt_full, dim=-1)).view(b, -1)
        log_x_t_full = torch.stack(log_x_t_full, dim=-1).view(b, self.num_classes, -1)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_x_t_full, t=t)  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_x_t_full, t=t
        )  # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start_full
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_x_t_full)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (
                x0_recon[index] == x0_real[index]
            ).sum().cpu() / x0_real.size()[1]
            self.diffusion_acc_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            )
            same_rate = (
                xt_1_recon[index] == xt_recon[index]
            ).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            )

        # compute log_true_prob now
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start_full, log_x_t=log_x_t_full, t=t
        )
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt_full == self.num_classes - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )
        kl = kl * mask_weight
        kl = mean_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start_full, log_model_prob)
        decoder_nll = mean_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1.0 - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        loss1 = kl_loss / pt
        losses = {"kl_loss": loss1.mean()}
        if self.auxiliary_loss_weight != 0 and is_train == True:
            kl_aux = self.multinomial_kl(
                log_x_start_full[:, :-1, :], log_x0_recon[:, :-1, :]
            )
            kl_aux = kl_aux * mask_weight
            kl_aux = mean_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt

            losses["aux_loss"] = loss2.mean()

        outputs = {"probs": log_model_prob.exp()}
        return outputs, losses
