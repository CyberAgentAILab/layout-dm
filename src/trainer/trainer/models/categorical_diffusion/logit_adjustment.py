import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
from torch import FloatTensor, Tensor
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.models.categorical_diffusion.util import index_to_log_onehot
from trainer.models.clg.const import relation as relational_constraints

logger = logging.getLogger(__name__)


def _stochastic_convert(
    cond: Dict,
    model_log_prob: Tensor,
    tokenizer: LayoutSequenceTokenizer,
    tau: float = 1.0,
    mode: str = "average",
) -> Tensor:
    """
    Convert model_log_prob (B, C, S) to average bbox location (E, X)
    , where E is number of valid layout components and X is number of fields in each component.
    Use mode='average' by default because 'gumbel' did not work at all.
    """
    assert mode in ["gumbel", "average"]
    B = model_log_prob.size(0)
    N = tokenizer.N_bbox_per_var
    device = model_log_prob.device
    step = len(tokenizer.var_names)
    bt = tokenizer.bbox_tokenizer

    # get bbox logits for canvas (B, C, X)
    canvas_ids = bt.encode(FloatTensor([[[0.5, 0.5, 1.0, 1.0]]])).long()
    canvas_ids += tokenizer.N_category
    canvas_logits = index_to_log_onehot(
        repeat(canvas_ids, "1 1 x -> b x", b=B), tokenizer.N_total
    ).to(model_log_prob)

    # get element-wise mask (B, S+1)
    mask = cond["seq"][..., ::step] != tokenizer.name_to_id("pad")
    mask = torch.cat([torch.full((B, 1), fill_value=True).to(mask), mask], dim=1).to(
        device
    )

    if bt.shared_bbox_vocab == "xywh":
        slices = [
            slice(tokenizer.N_category, tokenizer.N_category + N)
            for _ in range(step - 1)
        ]
    else:
        slices = [
            slice(tokenizer.N_category + i * N, tokenizer.N_category + (i + 1) * N)
            for i in range(step - 1)
        ]

    bbox_logits = []
    for i in range(step - 1):
        bbox_logit = torch.cat(
            [
                canvas_logits[:, slices[i], i : i + 1],
                model_log_prob[:, slices[i], (i + 1) :: step],
            ],
            dim=2,
        )
        # why requires_grad diminishes in maskgit?
        bbox_logits.append(bbox_logit)

    bbox_logits = rearrange(torch.stack(bbox_logits, dim=-1), "b n s x -> b s n x")
    bbox_logits = bbox_logits[mask]

    if mode == "gumbel":
        bbox_prob = F.gumbel_softmax(bbox_logits, tau=tau, hard=True, dim=1)
    elif mode == "average":
        bbox_prob = F.softmax(bbox_logits, dim=1)

    centers = []
    for name in bt.var_names:
        centers.append(bt.clustering_models[f"{name}-{N}"].cluster_centers_)
    centers = torch.cat([torch.from_numpy(arr) for arr in centers], dim=1)
    centers = rearrange(centers, "n x -> 1 n x")
    bbox = reduce(bbox_prob * centers.to(bbox_prob), "e n x -> e x", reduction="sum")
    return bbox


def update(
    t: int,
    cond: Dict,
    model_log_prob: FloatTensor,  # (B, C, S)
    tokenizer: LayoutSequenceTokenizer,
    sampling_cfg: Optional[DictConfig] = None,
):
    """
    Update model_log_prob multiple times following Eq. 7.
    model_log_prob corresponds to p_{\theta}(\bm{z}_{t-1}|\bm{z}_{t}).
    """
    # detach var. in order not to backpropagate thrhough diffusion model p_{\theta}.
    optim_target_log_prob = torch.nn.Parameter(model_log_prob.detach())

    # we found that adaptive optimizer does not work.
    optimizer = torch.optim.SGD(
        [optim_target_log_prob], lr=sampling_cfg.relation_lambda
    )
    batch = cond["batch_w_canvas"].to(model_log_prob.device)
    T = 0 if t < 10 else sampling_cfg.relation_num_update
    for _ in range(T):
        optimizer.zero_grad()
        bbox_flatten = _stochastic_convert(
            cond=cond,
            model_log_prob=optim_target_log_prob,
            tokenizer=tokenizer,
            tau=sampling_cfg.relation_tau,
            mode=sampling_cfg.relation_mode,
        )
        if len(batch.edge_index) == 0:
            # sometimes there are no edge in batch_size = 1
            continue
        loss = [f(bbox_flatten, batch) for f in relational_constraints]
        loss = torch.stack(loss, dim=-1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    return optim_target_log_prob.detach()
