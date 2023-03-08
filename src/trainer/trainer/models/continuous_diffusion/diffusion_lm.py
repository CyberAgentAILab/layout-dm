from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .base import ContinuousDiffusionBase


class DiffusionLM(ContinuousDiffusionBase):
    """
    Diffusion-LM Improves Controllable Text Generation (NeurIPS'22)
    https://arxiv.org/abs/2205.14217
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rounder = nn.Linear(self.num_channel, self.tokenizer.N_total)

    def dis2con(
        self, seq: LongTensor, reparametrize: bool = False, normalize: bool = False
    ) -> Union[FloatTensor, Tuple[FloatTensor, FloatTensor]]:
        """
        Args:
            seq: LongTensor with shape (B, S) indicating id of each token
        Returns:
            arr: FloatTensor with shape (B, S, D) indicating continuous vector
        """
        assert seq.dim() == 2
        emb = self.token_emb(seq)
        if normalize:
            emb = F.normalize(emb, dim=-1)
        if reparametrize and hasattr(self, "con2logits"):
            if hasattr(self, "scheduler"):  # w/ diffuser
                timestep = torch.zeros((1,), device=self.device).long()
                # get mean of the final distribution by setting zero input
                noise = self.scheduler.add_noise(
                    torch.zeros_like(emb), torch.randn_like(emb), timestep
                )
                emb_reparametrized = emb + noise
            else:
                from .base import log_snr_to_alpha_sigma

                rep_times = torch.zeros((1,), device=self.device).float()
                _, rep_sigma = log_snr_to_alpha_sigma(self.log_snr(rep_times))
                emb_reparametrized = emb + rep_sigma * torch.randn_like(emb)
            return emb_reparametrized, emb
        else:
            return emb

    def con2dis(self, arr: FloatTensor) -> LongTensor:
        """
        Args:
            arr: FloatTensor with shape (B, S, D) indicating continuous vector
        Returns:
            seq: LongTensor with shape (B, S) indicating id of each token
        """
        assert arr.dim() == 3
        seq = torch.argmax(self.con2logits(arr), dim=-1)
        return seq

    def con2logits(self, arr: FloatTensor) -> LongTensor:
        """
        Args:
            arr: FloatTensor with shape (B, S, D) indicating continuous vector
        Returns:
            logits: FloatTensor with shape (B, S, C) indicating logit for each discrete token
        """
        assert arr.dim() == 3
        logits = self.rounder(arr)
        return logits
