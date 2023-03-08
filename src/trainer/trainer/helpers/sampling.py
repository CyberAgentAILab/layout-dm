from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import FloatTensor, LongTensor

FILTER_VALUE = -float("Inf")


@dataclass
class DeterministicSamplingConfig:
    name: str = "deterministic"


@dataclass
class _StochasticSamplingConfig:
    temperature: float = 1.0


@dataclass
class RandomSamplingConfig(_StochasticSamplingConfig):
    name: str = "random"


@dataclass
class GumbelSamplingConfig(_StochasticSamplingConfig):
    name: str = "gumbel"


@dataclass
class TopKSamplingConfig(_StochasticSamplingConfig):
    name: str = "top_k"
    top_k: int = 5


@dataclass
class TopPSamplingConfig(_StochasticSamplingConfig):
    name: str = "top_p"
    top_p: float = 0.9


@dataclass
class TopKTopPSamplingConfig(_StochasticSamplingConfig):
    name: str = "top_k_top_p"
    top_k: int = 5
    top_p: float = 0.9


SAMPLING_CONFIG_DICT = {
    "top_k": TopKSamplingConfig,
    "top_k": TopKTopPSamplingConfig,
    "top_p": TopPSamplingConfig,
    "deterministic": DeterministicSamplingConfig,
    "random": RandomSamplingConfig,
    "gumbel": GumbelSamplingConfig,
}


def register_sampling_config(cs: ConfigStore):
    """
    Helper to register all sampling configurations defined above
    """
    cs.store(group="sampling", name="top_k", node=TopKSamplingConfig)
    cs.store(group="sampling", name="top_k_top_p", node=TopKTopPSamplingConfig)
    cs.store(group="sampling", name="top_p", node=TopPSamplingConfig)
    cs.store(group="sampling", name="deterministic", node=DeterministicSamplingConfig)
    cs.store(group="sampling", name="random", node=RandomSamplingConfig)


def top_k_logits(logits: FloatTensor, k: int, dim: int = -1):
    # logits: (B, C)
    v, _ = torch.topk(logits, k, dim)
    out = logits.clone()
    out[out < v[:, [-1]]] = FILTER_VALUE
    return out


def sample(logits: FloatTensor, sampling_cfg: DictConfig) -> LongTensor:
    """
    Input: logits (B, C, *N)
    Output: (B, 1, *N)
    """
    assert logits.ndim in [2, 3]
    if sampling_cfg.name == "deterministic":
        output = torch.argmax(logits, dim=1, keepdim=True)
    else:
        logits_ = logits / sampling_cfg.temperature

        if sampling_cfg.name == "top_k":
            logits = top_k_logits(logits_, k=sampling_cfg.top_k, dim=1)
        elif sampling_cfg.name == "top_p":
            top_p = sampling_cfg.top_p
            assert 0.0 < top_p <= 1.0

            S = logits.size(1)
            # https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort
            sorted_logits, sorted_indices = torch.sort(logits_, descending=True, dim=1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)

            indices = torch.arange(S).view(1, S).to(logits.device)
            if logits.ndim == 3:
                indices = indices.unsqueeze(dim=-1)

            # make sure to keep the first logit (most likely one)
            sorted_logits[(cumulative_probs > top_p) & (indices > 0)] = FILTER_VALUE
            logits = sorted_logits.gather(dim=1, index=sorted_indices.argsort(dim=1))
        elif sampling_cfg.name == "random":
            logits = logits_
        elif sampling_cfg.name == "gumbel":
            uniform = torch.rand_like(logits_)
            const = 1e-30
            gumbel_noise = -torch.log(-torch.log(uniform + const) + const)
            logits = logits_ + gumbel_noise
        else:
            raise NotImplementedError

        probs = F.softmax(logits, dim=1)
        if probs.ndim == 2:
            output = torch.multinomial(probs, num_samples=1)  # (B, 1)
        elif probs.ndim == 3:
            S = probs.shape[2]
            probs = rearrange(probs, "b c s -> (b s) c")
            output = torch.multinomial(probs, num_samples=1)
            output = rearrange(output, "(b s) 1 -> b 1 s", s=S)
        else:
            raise NotImplementedError
    return output


if __name__ == "__main__":
    from einops import repeat
    from omegaconf import OmegaConf

    sampling_cfg = OmegaConf.create({"name": "top_p", "top_p": 0.9, "temperature": 1.0})
    logits = repeat(torch.arange(5), "c -> b c 1", b=2)
    x = sample(logits, sampling_cfg, return_confidence=True)
    print(x)
