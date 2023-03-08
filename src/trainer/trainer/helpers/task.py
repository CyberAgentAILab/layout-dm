import random
from typing import Any, Dict

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
from torch import LongTensor, Tensor
from torch_geometric.utils import to_dense_batch
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.mask import sample_mask
from trainer.helpers.util import batch_topk_mask

MAX_PARTIAL_RATIO = 0.3
COND_TYPES = [
    "c",  # given category, predict position/size (C->P+S)
    "cwh",  # given category/size, predict position (C+S->P)
    "partial",  # given a partial layout (i.e., only a few elements), predict to generate a complete layout
    "gt",  # just copy
    "random",  # random masking
    "refinement",  # given category and noisy position/size, predict accurate position/size
    "relation",  # given category and some relationships between elements, try to fulfill the relationships as much as possible
]


def get_cond(
    batch,  # torch_geometric.data.batch.DataBatch
    tokenizer: LayoutSequenceTokenizer,
    cond_type: str = "c",
    model_type: str = "",
    get_real_images: bool = False,
) -> Dict[str, Any]:
    assert cond_type in COND_TYPES

    if get_real_images:
        assert cond_type in ["cwh", "gt"]

    special_keys = tokenizer.special_tokens
    pad_id = tokenizer.name_to_id("pad")
    mask_id = tokenizer.name_to_id("mask") if "mask" in special_keys else -1

    if cond_type == "relation":
        # extract non-canvas variables
        flag = batch.attr["has_canvas_element"]
        if isinstance(flag, bool):
            assert flag
        elif isinstance(flag, Tensor):
            assert flag.all()
        remove_canvas = True
    else:
        remove_canvas = False

    # load real layouts
    bbox, label, _, mask = sparse_to_dense(batch, remove_canvas=remove_canvas)
    cond = tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
    B = bbox.shape[0]
    S = cond["seq"].shape[1]
    C = tokenizer.N_var_per_element

    # modify some elements to simulate various conditional generation settings
    if cond_type == "partial":
        start = 1 if "bos" in special_keys else 0
        n_elem = (S - start) // C
        scores = torch.rand(B, n_elem)
        mask = cond["mask"][:, start::C]

        n_valid_elem = reduce(mask, "b s -> b", reduction="sum")
        topk = []
        for k in n_valid_elem:
            vmax = int((k - 1) * MAX_PARTIAL_RATIO)
            val = random.randint(1, vmax) if vmax > 1 else 1
            topk.append(val)
        topk = torch.LongTensor(topk)
        keep, _ = batch_topk_mask(scores, topk, mask=mask)

        keep = repeat(keep, "b s -> b (s c)", c=C)
        if "bos" in special_keys:
            # for order-sensitive methods, shift valid condition at the beginning of seq.
            keep = torch.cat([torch.full((B, 1), fill_value=True), keep], dim=-1)
            new_seq = torch.full_like(cond["seq"], mask_id)
            new_mask = torch.full_like(cond["mask"], False)
            for i in range(B):
                s = cond["seq"][i]
                ind_end = keep[i].sum().item()
                new_seq[i][:ind_end] = s[keep[i]]
                new_mask[i][:ind_end] = True
            cond["seq"] = new_seq
            cond["mask"] = new_mask
        else:
            cond["seq"][~keep] = mask_id
            cond["mask"] = keep

    elif cond_type in ["c", "cwh", "relation"]:
        vars = {"c": "c", "cwh": "cwh", "relation": "c"}
        keep = torch.full((B, S), False)
        if "bos" in special_keys:
            attr_ind = (torch.arange(S).view(1, S) - 1) % C
            attr_ind[:, 0] = -1  # dummy id for BOS
            keep[:, 0] = True
        else:
            attr_ind = torch.arange(S).view(1, S) % C
        for s in vars[cond_type]:
            ind = tokenizer.var_names.index(s)
            keep |= attr_ind == ind
        cond["seq"][~keep] = mask_id

        # specifying number of elements since it is known in these settings
        cond["seq"][~cond["mask"]] = pad_id
        cond["mask"] = (cond["mask"] & keep) | ~cond["mask"]

        # load edge attributes for imposing relational constraints
        if cond_type == "relation":
            cond["batch_w_canvas"] = batch

    elif cond_type == "gt":
        pass

    elif cond_type == "random":
        ratio = torch.rand((B,))
        loss_mask = sample_mask(torch.full(cond["mask"].size(), True), ratio)
        # pass
        cond["seq"][loss_mask] = mask_id
        cond["mask"] = ~loss_mask

    elif cond_type == "refinement":
        new_bbox = bbox + torch.normal(0, std=0.1, size=bbox.size())
        new_cond = tokenizer.encode({"label": label, "mask": mask, "bbox": new_bbox})
        index = repeat(torch.arange(S), "s -> b s", b=B)
        cond = {}
        if "bos" in special_keys:
            cond["mask"] = new_cond["mask"] & ((index - 1) % C == 0) | ~new_cond["mask"]
        else:
            cond["mask"] = new_cond["mask"] & (index % C == 0) | ~new_cond["mask"]
        if model_type in ["LayoutDM", "ElemWiseAutoreg"]:
            cond["seq"] = torch.where(cond["mask"], new_cond["seq"], mask_id)
            cond["seq"] = torch.where(new_cond["mask"], cond["seq"], pad_id)
            cond["seq_orig"] = new_cond["seq"]
        else:
            cond["seq"] = new_cond["seq"]
    else:
        raise NotImplementedError

    if get_real_images:
        pass

    cond["type"] = cond_type
    if cond_type in ["c", "cwh", "refinement", "relation"]:
        cond["num_element"] = mask.sum(dim=1)

    return cond


def _index_to_smoothed_log_onehot(
    seq: LongTensor,
    tokenizer: LayoutSequenceTokenizer,
    mode: str = "uniform",
    offset_ratio: float = 0.2,
):
    # for ease of hp-tuning, the range is limited to [0.0, 1.0]
    assert tokenizer.N_var_per_element == 5
    assert mode in ["uniform", "gaussian", "negative"]

    bbt = tokenizer.bbox_tokenizer
    V = len(bbt.var_names)
    N = tokenizer.N_bbox_per_var

    if tokenizer.bbox_tokenizer.shared_bbox_vocab == "xywh":
        slices = [
            slice(tokenizer.N_category, tokenizer.N_category + N) for i in range(V)
        ]
    else:
        slices = [
            slice(tokenizer.N_category + i * N, tokenizer.N_category + (i + 1) * N)
            for i in range(V)
        ]

    logits = torch.zeros(
        (tokenizer.N_total, tokenizer.N_total),
    )
    logits.fill_diagonal_(1.0)

    for i, key in enumerate(bbt.var_names):
        name = f"{key}-{N}"
        cluster_model = tokenizer.bbox_tokenizer.clustering_models[name]
        cluster_centers = torch.from_numpy(cluster_model.cluster_centers_).view(-1)
        ii, jj = torch.meshgrid(cluster_centers, cluster_centers, indexing="ij")
        if mode == "uniform":
            logits[slices[i], slices[i]] = (torch.abs(ii - jj) < offset_ratio).float()
        elif mode == "negative":
            logits[slices[i], slices[i]] = (torch.abs(ii - jj) >= offset_ratio).float()
        elif mode == "gaussian":
            # p(x) = a * exp( -(x-b)^2 / (2 * c^2))
            # -> log p(x) = log(a) - (x-b)^2 / (2 * c^2)
            # thus, a strength of adjustment is proportional to -(ii - jj)^2
            logits[slices[i], slices[i]] = -1.0 * (ii - jj) ** 2
        else:
            raise NotImplementedError

    logits = rearrange(F.embedding(seq, logits), "b s c -> b c s")
    return logits


def set_additional_conditions_for_refinement(
    cond: Dict[str, Any],
    tokenizer: LayoutSequenceTokenizer,
    sampling_cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Set hand-crafted prior for the position/size of each element (Eq. 8)
    """
    w = sampling_cfg.refine_lambda
    if sampling_cfg.refine_mode == "negative":
        w *= -1.0

    cond["weak_mask"] = repeat(~cond["mask"], "b s -> b c s", c=tokenizer.N_total)
    cond["weak_logits"] = _index_to_smoothed_log_onehot(
        cond["seq_orig"],
        tokenizer,
        mode=sampling_cfg.refine_mode,
        offset_ratio=sampling_cfg.refine_offset_ratio,
    )
    cond["weak_logits"] *= w
    return cond


def filter_canvas(layout: Dict):
    new_layout = {}
    new_layout["bbox"] = layout["bbox"][:, 1:]
    new_layout["label"] = layout["label"][:, 1:] - 1
    new_layout["mask"] = layout["mask"][:, 1:]
    return new_layout


def duplicate_cond(cond: Dict, batch_size: int) -> Dict:
    # this is used in demo to see the variety
    # if there's single example but batch_size > 1, copy conditions
    flag = cond["seq"].size(0) == 1
    flag &= batch_size > 1
    if flag:
        for k in cond:
            if isinstance(cond[k], Tensor):
                sizes = [
                    batch_size,
                ]
                sizes += [1 for _ in range(cond[k].dim() - 1)]
                cond[k] = cond[k].repeat(sizes)
    return cond
