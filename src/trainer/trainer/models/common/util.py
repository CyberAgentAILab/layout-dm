import copy
import logging
import os

import fsspec
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import BoolTensor

logger = logging.getLogger(__name__)


def generate_causal_mask(sz: int) -> BoolTensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def get_dim_model(backbone_cfg: DictConfig) -> int:
    """
    It takes hierarchical config for
    - trainer.models.transformer_utils import TransformerEncoder
    and get a number of dimension inside the Transformer
    """
    result = None
    for key, value in backbone_cfg.items():
        if key == "d_model":
            result = value
        elif isinstance(value, DictConfig):
            x = get_dim_model(value)
            if x:
                result = x
    return result


def shrink(backbone_cfg: DictConfig, mult: float) -> DictConfig:
    """
    Rescale dimension of a model linearly
    """
    new_backbone_cfg = copy.deepcopy(backbone_cfg)
    l = new_backbone_cfg.encoder_layer
    for key in ["d_model", "dim_feedforward"]:
        new_backbone_cfg.encoder_layer[key] = int(mult * l[key])
    return new_backbone_cfg


def load_model(
    model: nn.Module,
    ckpt_dir: str,
    device: torch.device,
    best_or_final: str = "best",
    extension: str = ".pt",
):
    model_path = os.path.join(ckpt_dir, f"{best_or_final}_model{extension}")
    with fsspec.open(str(model_path), "rb") as file_obj:
        model.load_state_dict(torch.load(file_obj, map_location=device))
    return model


def save_model(model: nn.Module, ckpt_dir: str, best_or_final: str = "best"):
    model_path = os.path.join(ckpt_dir, f"{best_or_final}_model.pt")
    with fsspec.open(str(model_path), "wb") as file_obj:
        torch.save(model.state_dict(), file_obj)
    return model
