import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from trainer.models.transformer_utils import TransformerEncoder

from .layout import LayoutDecoder, LayoutEncoder
from .util import generate_causal_mask

logger = logging.getLogger(__name__)


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        # https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class SeqLengthDistribution(nn.Module):
    def __init__(self, max_seq_length: int, weight=0.999) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.weight = weight

        # logger.warning("EMA for seq_length is computed during training")
        fill_value = 1 / max_seq_length
        self.register_buffer(
            "n_elements_prob",
            torch.full((max_seq_length,), fill_value=fill_value),
        )

    def __call__(self, mask: BoolTensor):
        N = self.max_seq_length
        batch_prob = mask.sum(dim=1).bincount(minlength=N + 1)[1:] / mask.size(0)
        self.n_elements_prob = self.weight * self.n_elements_prob
        self.n_elements_prob += (1.0 - self.weight) * batch_prob.to(
            self.n_elements_prob
        )

    def sample(self, batch_size: int) -> LongTensor:
        n_elements = torch.multinomial(
            self.n_elements_prob.cpu(), batch_size, replacement=True
        )
        n_elements += 1  # shoule be in range [1, cfg.dataset.max_seq_length]
        return n_elements


class VAEModule(nn.Module):
    def __init__(self, dim_input: int, dim_latent: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(dim_input, dim_latent)
        self.fc_var = nn.Linear(dim_input, dim_latent)

    def reparameterize(self, mu: FloatTensor, logvar: FloatTensor) -> FloatTensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: FloatTensor) -> Dict[str, FloatTensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        return {"z": z, "mu": mu, "logvar": logvar}


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, max_token_length: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.rand(max_token_length, dim_model))

    def forward(self, h: Tensor):
        B, S = h.shape[:2]
        emb = rearrange(self.pos_emb[:S], "s d -> 1 s d")
        emb = repeat(emb, "1 s d -> b s d", b=B)
        return emb

    @property
    def no_decay_param_names(self) -> List[str]:
        return [
            "pos_emb",
        ]


class ElementPositionalEmbedding(torch.nn.Module):
    """
    Positional embedding to indicate j-th attr of i-th element
    """

    def __init__(self, dim_model: int, max_token_length: int, n_attr_per_elem=5):
        super().__init__()
        remainder = max_token_length % n_attr_per_elem
        if remainder == 1:
            self.bos_emb = nn.Parameter(torch.rand(1, dim_model))
        elif remainder == 0:
            pass
        else:
            raise NotImplementedError

        self.max_len = max_token_length
        self.n_elem = max_token_length // n_attr_per_elem
        self.n_attr_per_elem = n_attr_per_elem
        self.elem_emb = nn.Parameter(torch.rand(self.n_elem, dim_model))
        self.attr_emb = nn.Parameter(torch.rand(self.n_attr_per_elem, dim_model))

    def forward(self, h: Tensor):
        if getattr(self, "bos_emb", None) is not None:
            h = h[:, 1:]
        B, S = h.size()[:2]

        # (1, 2, 3) -> (1, ..., 1, 2, ..., 2, 3, ..., 3, ...)
        elem_emb = repeat(self.elem_emb, "s d -> (s x) d", x=self.n_attr_per_elem)
        # (1, 2, 3) -> (1, 2, 3, 1, 2, 3, ...)
        attr_emb = repeat(self.attr_emb, "x d -> (s x) d", s=self.n_elem)
        emb = elem_emb + attr_emb

        emb = emb[:S]
        if getattr(self, "bos_emb", None) is not None:
            emb = torch.cat([self.bos_emb, emb], dim=0)
        emb = repeat(emb, "s d -> b s d", b=B)
        return emb

    @property
    def no_decay_param_names(self) -> List[str]:
        decay_list = ["elem_emb", "attr_emb"]
        if getattr(self, "bos_emb", None) is not None:
            decay_list.append("bos_emb")
        return decay_list


class CategoricalTransformer(torch.nn.Module):
    """
    Model to perform one-shot / auto-regressive generation of 1D sequence
    """

    def __init__(
        self,
        # backbone_cfg: DictConfig,
        backbone: TransformerEncoder,
        num_classes: int,
        max_token_length: int,
        dim_model: int,
        lookahead: bool = True,
        pos_emb: str = "default",
        dim_head: Optional[int] = None,
        use_additional_input: Optional[str] = None,  # for self-conditioned generation
        **kwargs,
    ) -> None:
        super().__init__()

        self.lookahead = lookahead
        self.use_additional_input = use_additional_input
        self.backbone = backbone
        self.cat_emb = nn.Embedding(num_classes, dim_model)

        if self.use_additional_input == "seq":
            self.cat_emb_additional = nn.Embedding(num_classes, dim_model)
        elif self.use_additional_input in ["logit", "prob"]:
            self.cat_emb_additional = nn.Linear(num_classes, dim_model)
        if self.use_additional_input:
            self.self_cond_fuser = nn.Sequential(
                nn.Linear(dim_model * 2, dim_model),
                nn.ReLU(),
            )

        if pos_emb == "default":
            self.pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length
            )
        elif pos_emb == "elem_attr":
            self.pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
        else:
            raise NotImplementedError

        self.drop = nn.Dropout(0.1)
        d_last = dim_head if dim_head else num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, d_last, bias=False)
        )

    def forward(
        self,
        seq: Union[Tensor, Tuple[Tensor, Tensor]],
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S)
        Output: 2D sequence of logits (B, S, C)
        """

        S = seq.shape[1]
        h = self.cat_emb(seq)

        if self.use_additional_input:
            if self_cond is not None:
                if self.use_additional_input == "seq":
                    h_add = self.cat_emb_additional(self_cond)
                elif self.use_additional_input in ["logit", "prob"]:
                    h_add = self.cat_emb_additional(
                        rearrange(self_cond, "b c s -> b s c")
                    )
                else:
                    raise NotImplementedError
            else:
                h_add = torch.zeros_like(h)
            h = self.self_cond_fuser(torch.cat([h, h_add], dim=-1))

        h = h + self.pos_emb(seq)
        h = self.drop(h)

        if self.lookahead:
            # non-autoregressive generation
            if timestep is not None:
                h = self.backbone(
                    h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
                )
            else:
                h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # autoregressive generation
            mask = generate_causal_mask(S).to(h)
            h = self.backbone(h, mask=mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(h)  # (B, S, C)
        outputs = {"logits": logits}
        return outputs


class ContinuousTransformer(torch.nn.Module):
    """
    Model to perform one-shot / auto-regressive generation of 1D sequence (B, S, C)
    """

    def __init__(
        self,
        backbone: TransformerEncoder,
        max_token_length: int,
        dim_model: int,
        dim_in: int,
        lookahead: bool = True,
        pos_emb: str = "default",
        # use_additional_input: Optional[str] = None,  # for self-conditioned generation
        **kwargs,
    ) -> None:
        super().__init__()

        self.lookahead = lookahead
        self.backbone = backbone
        self.emb = nn.Linear(dim_in * 2, dim_model)

        if pos_emb == "default":
            self.pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length
            )
        elif pos_emb == "elem_attr":
            self.pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
        else:
            raise NotImplementedError

        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, dim_in, bias=False)
        )

    def forward(
        self,
        x: FloatTensor,
        src_key_padding_mask: Optional[BoolTensor] = None,
        timestep: Optional[Union[LongTensor, FloatTensor]] = None,
        x_self_cond: Optional[FloatTensor] = None,
    ) -> Tensor:
        """
        Input: 2D sequence of shape (B, S, C)
        Output: 2D sequence of logits (B, S, C)
        """
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x)
        x = torch.cat((x_self_cond, x), dim=-1)

        h = self.emb(x)
        h = h + self.pos_emb(h)
        h = self.drop(h)

        if self.lookahead:
            # non-autoregressive generation
            if timestep is not None:
                h = self.backbone(
                    h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
                )
            else:
                h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # autoregressive generation
            # mask = generate_causal_mask(S).to(h)
            # h = self.backbone(h, mask=mask, src_key_padding_mask=src_key_padding_mask)
            raise NotImplementedError
        outputs = {"outputs": self.head(h)}  # (B, S, C)
        return outputs


class CategoricalEncDecTransformer(torch.nn.Module):
    """
    For bart-like models
    """

    def __init__(
        self,
        backbone_enc: TransformerEncoder,
        backbone_dec: nn.TransformerDecoder,
        num_classes_dec: int,
        max_token_length_dec: int,
        dim_model: int,
        pos_emb: str = "default",
        dim_head: Optional[int] = None,
        num_classes_enc: Optional[int] = None,
        max_token_length_enc: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = backbone_enc
        self.decoder = backbone_dec

        if num_classes_enc is None:
            num_classes_enc = num_classes_dec
        if max_token_length_enc is None:
            max_token_length_enc = max_token_length_dec

        self.input_cat_emb = nn.Embedding(num_classes_enc, dim_model)
        self.target_cat_emb = nn.Embedding(num_classes_dec, dim_model)

        if pos_emb == "default":
            self.input_pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length_enc
            )
            self.target_pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length_dec
            )
        elif pos_emb == "elem_attr":
            self.input_pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length_enc,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
            self.target_pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length_dec,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
        else:
            raise NotImplementedError

        self.drop = nn.Dropout(0.1)
        d_last = dim_head if dim_head else num_classes_dec
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, d_last, bias=False)
        )

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S), first token is always [BOS]
        Output: 2D sequence of logits (B, S, C)
        """

        h_enc = self.input_cat_emb(input)
        h_enc += self.input_pos_emb(input)

        h_enc = self.drop(h_enc)
        memory = self.encoder(h_enc, src_key_padding_mask=src_key_padding_mask)

        tgt = self.target_cat_emb(target)
        tgt += self.target_pos_emb(target)

        S = target.shape[1]
        tgt_mask = generate_causal_mask(S).to(tgt)
        h = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        logits = self.head(h)  # (B, S, C)
        outputs = {"logits": logits}
        return outputs


class CategoricalAggregatedTransformer(CategoricalTransformer):
    """
    Model to perform one-shot / auto-regressive generation of 1D sequence
    """

    def __init__(
        self,
        backbone_cfg: DictConfig,
        num_classes: int,
        max_len: int,
        lookahead: bool = True,
    ) -> None:
        super().__init__(backbone_cfg, num_classes, max_len, lookahead)
        assert self.lookahead
        self.enc = nn.Sequential(
            nn.Linear(5 * self.d_model, self.d_model),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.d_model, 5 * self.d_model),
            nn.ReLU(),
        )

    def forward(
        self,
        seq: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S)
        Output: 2D sequence of logits (B, S, C)
        """
        # (guess): positional info. should be added just before transformer blocks
        h = self.drop(self.cat_emb(seq))
        h = rearrange(h, "b (s x) d -> b s (x d)", x=5)
        h = self.enc(h)
        h = h + self.pos_emb(h)

        if timestep is not None:
            h = self.backbone(
                h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
            )
        else:
            h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)

        h = self.dec(h)
        h = rearrange(h, "b s (x d) -> b (s x) d ", x=5)
        outputs = {"logits": self.head(h)}  # (B, S, C)
        return outputs


class ElementTransformer(torch.nn.Module):
    """
    Model to perform one-shot / auto-regressive generation of elemtns
    """

    def __init__(
        self,
        # backbone_cfg: DictConfig,
        backbone: TransformerEncoder,
        num_labels: int,
        num_bin_bboxes: int,
        max_len: int,
        dim_model: int,
        lookahead: bool = False,
    ) -> None:
        super().__init__()
        # d_model = get_dim_model(backbone_cfg)
        self.backbone = backbone

        self.lookahead = lookahead

        self.layout_enc = LayoutEncoder(dim_model, num_labels, num_bin_bboxes)
        self.layout_dec = LayoutDecoder(dim_model, num_labels, num_bin_bboxes)

        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S)
        Output: 2D sequence of logits (B, S, C)
        """
        h = self.layout_enc(src)
        h = self.drop(h)

        if self.lookahead:
            # non-autoregressive generation
            if timestep is not None:
                h = self.backbone(
                    h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
                )
            else:
                h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # autoregressive generation
            mask = generate_causal_mask(h.size(1)).to(h)
            h = self.backbone(h, mask=mask, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        outputs = self.layout_dec(h)
        return outputs
