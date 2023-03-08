import logging

import torch
from omegaconf import DictConfig

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


class VanillaMaskAndReplaceDiffusion(BaseMaskAndReplaceDiffusion):
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
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_cfg=backbone_cfg,
            num_classes=num_classes,
            max_token_length=max_token_length,
            num_timesteps=num_timesteps,
            **kwargs,
        )

        if self.alpha_init_type == "alpha1":
            N = self.num_classes - 1
            at, bt, ct, att, btt, ctt = self.alpha_schedule_partial_func(N=N)
        else:
            print("alpha_init_type is Wrong !! ")

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
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_pred(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(
            self.log_1_min_cumprod_ct, t, log_x_start.shape
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
        self, log_x_start, log_x_t, t
    ):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(
            -1, -1, self.max_token_length
        )

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = (
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        )
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def forward(self, x, is_train=True):  # get the KL loss
        b, device = x.size(0), x.device

        assert self.loss_type == "vb_stochastic"
        x_start = x
        t, pt = self.sample_time(b, device, "importance")

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, t=t)  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_xt, t=t
        )  # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
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
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )
        kl = kl * mask_weight
        kl = mean_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
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
                log_x_start[:, :-1, :], log_x0_recon[:, :-1, :]
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
