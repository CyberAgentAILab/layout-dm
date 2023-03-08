import multiprocessing
from functools import partial
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as tdist
from einops import rearrange, reduce, repeat
from prdc import compute_prdc
from pytorch_fid.fid_score import calculate_frechet_distance
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from torch import BoolTensor, FloatTensor
from torch_geometric.utils import to_dense_adj
from trainer.data.util import RelLoc, RelSize, detect_loc_relation, detect_size_relation
from trainer.helpers.util import convert_xywh_to_ltrb

Feats = Union[FloatTensor, List[FloatTensor]]
Layout = Tuple[np.ndarray, np.ndarray]

# set True to disable parallel computing by multiprocessing (typically for debug)
# DISABLED = False
DISABLED = True


def __to_numpy_array(feats: Feats) -> np.ndarray:
    if isinstance(feats, list):
        # flatten list of batch-processed features
        if isinstance(feats[0], FloatTensor):
            feats = [x.detach().cpu().numpy() for x in feats]
    else:
        feats = feats.detach().cpu().numpy()
    return np.concatenate(feats)


def compute_generative_model_scores(
    feats_real: Feats,
    feats_fake: Feats,
) -> Dict[str, float]:
    """
    Compute precision, recall, density, coverage, and FID.
    """
    feats_real = __to_numpy_array(feats_real)
    feats_fake = __to_numpy_array(feats_fake)

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    results = compute_prdc(
        real_features=feats_real, fake_features=feats_fake, nearest_k=5
    )
    results["fid"] = calculate_frechet_distance(
        mu_real, sigma_real, mu_fake, sigma_fake
    )

    return results


def compute_violation(bbox_flatten, data):
    """
    Compute relation violation accuracy as in LayoutGAN++ [Kikuchi+, ACMMM'21].
    """
    device = data.x.device
    failures, valid = [], []

    _zip = zip(data.edge_attr, data.edge_index.t())
    for gt, (i, j) in _zip:
        failure, _valid = 0, 0
        b1, b2 = bbox_flatten[i], bbox_flatten[j]

        # size relation
        if ~gt & 1 << RelSize.UNKNOWN:
            pred = detect_size_relation(b1, b2)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        # loc relation
        if ~gt & 1 << RelLoc.UNKNOWN:
            canvas = data.y[i].eq(0)
            pred = detect_loc_relation(b1, b2, canvas)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        failures.append(failure)
        valid.append(_valid)

    failures = torch.as_tensor(failures).to(device)
    failures = to_dense_adj(data.edge_index, data.batch, failures)
    valid = torch.as_tensor(valid).to(device)
    valid = to_dense_adj(data.edge_index, data.batch, valid)

    return failures.sum((1, 2)) / valid.sum((1, 2))


def compute_alignment(bbox: FloatTensor, mask: BoolTensor) -> Dict[str, FloatTensor]:
    """
    Computes some alignment metrics that are different to each other in previous works.
    Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """
    S = bbox.size(1)

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log(1 - X)

    # original
    # return X.sum(-1) / mask.float().sum(-1)

    score = reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    Y = torch.stack([xl, xc, xr], dim=1)
    Y = rearrange(Y, "b x s -> b x 1 s") - rearrange(Y, "b x s -> b x s 1")

    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=Y.device)
    batch_mask[:, idx, idx] = True
    batch_mask = repeat(batch_mask, "b s1 s2 -> b x s1 s2", x=3)
    Y[batch_mask] = 1.0

    # Y = rearrange(Y.abs(), "b x s1 s2 -> b s1 x s2")
    # Y = reduce(Y, "b x s1 s2 -> b x", "min")
    # Y = rearrange(Y.abs(), " -> b s1 x s2")
    Y = reduce(Y.abs(), "b x s1 s2 -> b s1", "min")
    Y[Y == 1.0] = 0.0
    score_Y = reduce(Y, "b s -> b", "sum")

    results = {
        "alignment-ACLayoutGAN": score,
        "alignment-LayoutGAN++": score_normalized,
        "alignment-NDN": score_Y,
    }
    return results


def compute_overlap(bbox: FloatTensor, mask: BoolTensor) -> Dict[str, FloatTensor]:
    """
    Based on
    (i) Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    (ii) LAYOUTGAN: GENERATING GRAPHIC LAYOUTS WITH WIREFRAME DISCRIMINATORS (ICLR2019)
    https://arxiv.org/abs/1901.06767
    "percentage of total overlapping area among any two bounding boxes inside the whole page."
    At least BLT authors seems to sum. (in the MSCOCO case, it surpasses 1.0)
    """
    B, S = mask.size()
    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    # diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    # ai = ai.masked_fill(diag_mask, 0)
    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=ai.device)
    batch_mask[:, idx, idx] = True
    ai = ai.masked_fill(batch_mask, 0)

    ar = torch.nan_to_num(ai / a1)  # (B, S, S)

    # original
    # return ar.sum(dim=(1, 2)) / mask.float().sum(-1)

    # fixed to avoid the case with single bbox
    score = reduce(ar, "b s1 s2 -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    ids = torch.arange(S)
    ii, jj = torch.meshgrid(ids, ids, indexing="ij")
    ai[repeat(ii >= jj, "s1 s2 -> b s1 s2", b=B)] = 0.0
    overlap = reduce(ai, "b s1 s2 -> b", reduction="sum")

    results = {
        "overlap-ACLayoutGAN": score,
        "overlap-LayoutGAN++": score_normalized,
        "overlap-LayoutGAN": overlap,
    }
    return results


def compute_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
    generalized: bool = False,
) -> Union[np.ndarray, FloatTensor]:
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou


def compute_perceptual_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
) -> Union[np.ndarray, FloatTensor]:
    """
    Computes 'Perceptual' IoU [Kong+, BLT'22]
    """
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    # numpy-only procedure in this part
    if isinstance(box_1, FloatTensor):
        unique_box_1 = np.unique(box_1.numpy(), axis=0)
    else:
        unique_box_1 = np.unique(box_1, axis=0)
    N = 32
    l1, t1, r1, b1 = [
        (x * N).round().astype(np.int32).clip(0, N)
        for x in convert_xywh_to_ltrb(unique_box_1.T)
    ]
    canvas = np.zeros((N, N))
    for (l, t, r, b) in zip(l1, t1, r1, b1):
        canvas[t:b, l:r] = 1
    global_area_union = canvas.sum() / (N**2)

    if global_area_union > 0.0:
        iou = ai / global_area_union
        return iou
    else:
        return lib.zeros((1,))


def __compute_maximum_iou_for_layout(layout_1: Layout, layout_2: Layout) -> float:
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        # note: maximize is supported only when scipy >= 1.4
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2: Tuple[List[Layout]]) -> np.ndarray:
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list: List[Layout]) -> Dict[str, List[Layout]]:
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
    disable_parallel: bool = DISABLED,
    n_jobs: Optional[int] = None,
):
    """
    Computes Maximum IoU [Kikuchi+, ACMMM'21]
    """
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    # to check actual number of layouts for evaluation
    # ans = 0
    # for x in args:
    #     ans += len(x[0])
    if disable_parallel:
        scores = [__compute_maximum_iou(a) for a in args]
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    if len(scores) == 0:
        return 0.0
    else:
        return scores.mean().item()


def __compute_average_iou(layout: Layout, perceptual: bool = False) -> float:
    bbox, _ = layout
    N = bbox.shape[0]
    if N in [0, 1]:
        return 0.0  # no overlap in principle

    ii, jj = np.meshgrid(range(N), range(N))
    ii, jj = ii.flatten(), jj.flatten()
    is_non_diag = ii != jj  # IoU for diag is always 1.0
    ii, jj = ii[is_non_diag], jj[is_non_diag]

    if perceptual:
        iou = compute_perceptual_iou(bbox[ii], bbox[jj])
    else:
        iou = compute_iou(bbox[ii], bbox[jj])

    # pick all pairs of overlapped objects
    cond = iou > np.finfo(np.float32).eps  # to avoid very-small nonzero
    # return iou.mean().item()
    if len(iou[cond]) > 0:
        return iou[cond].mean().item()
    else:
        return 0.0


def compute_average_iou(
    layouts: List[Layout],
    disable_parallel: bool = DISABLED,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute IoU between overlapping objects for each layout.
    Note that the lower is better unlike popular IoU.

    Reference:
        Variational Transformer Networks for Layout Generation (CVPR2021)
        https://arxiv.org/abs/2104.02416
    Reference: (perceptual version)
        BLT: Bidirectional Layout Transformer for Controllable Layout Generation (ECCV2022)
        https://arxiv.org/abs/2112.05112
    """
    func1 = partial(__compute_average_iou, perceptual=True)
    func2 = partial(__compute_average_iou, perceptual=False)

    # single-thread process for debugging
    if disable_parallel:
        scores1 = [func1(l) for l in layouts]
        scores2 = [func2(l) for l in layouts]
    else:
        with multiprocessing.Pool(n_jobs) as p1:
            scores1 = p1.map(func1, layouts)
        with multiprocessing.Pool(n_jobs) as p2:
            scores2 = p2.map(func2, layouts)
    results = {
        "average_iou-BLT": np.array(scores1).mean().item(),
        "average_iou-VTN": np.array(scores2).mean().item(),
    }
    return results


def __compute_bbox_sim(
    bboxes_1: np.ndarray,
    category_1: np.int64,
    bboxes_2: np.ndarray,
    category_2: np.int64,
    C_S: float = 2.0,
    C: float = 0.5,
) -> float:
    # bboxes from diffrent categories never match
    if category_1 != category_2:
        return 0.0

    cx1, cy1, w1, h1 = bboxes_1
    cx2, cy2, w2, h2 = bboxes_2

    delta_c = np.sqrt(np.power(cx1 - cx2, 2) + np.power(cy1 - cy2, 2))
    delta_s = np.abs(w1 - w2) + np.abs(h1 - h2)
    area = np.minimum(w1 * h1, w2 * h2)
    alpha = np.power(np.clip(area, 0.0, None), C)

    weight = alpha * np.power(2.0, -1.0 * delta_c - C_S * delta_s)
    return weight


def __compute_docsim_between_two_layouts(
    layouts_1_layouts_2: Tuple[List[Layout]],
    max_diff_thresh: int = 3,
) -> float:
    layouts_1, layouts_2 = layouts_1_layouts_2
    bboxes_1, categories_1 = layouts_1
    bboxes_2, categories_2 = layouts_2

    N, M = len(bboxes_1), len(bboxes_2)
    if N >= M + max_diff_thresh or N <= M - max_diff_thresh:
        return 0.0

    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_bbox_sim(
                bboxes_1[i], categories_1[i], bboxes_2[j], categories_2[j]
            )
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)

    if len(scores[ii, jj]) == 0:
        # sometimes, predicted bboxes are somehow filtered.
        return 0.0
    else:
        return scores[ii, jj].mean()


def compute_docsim(
    layouts_gt: List[Layout],
    layouts_generated: List[Layout],
    disable_parallel: bool = DISABLED,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute layout-to-layout similarity and average over layout pairs.
    Note that this is different from layouts-to-layouts similarity.
    """
    args = list(zip(layouts_gt, layouts_generated))
    if disable_parallel:
        scores = []
        for arg in args:
            scores.append(__compute_docsim_between_two_layouts(arg))
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_docsim_between_two_layouts, args)
    return np.array(scores).mean()


def _compute_wasserstein_distance_class(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
    n_categories: int = 25,
) -> float:
    categories_1 = np.concatenate([l[1] for l in layouts_1])
    counts = np.array(
        [np.count_nonzero(categories_1 == i) for i in range(n_categories)]
    )
    prob_1 = counts / np.sum(counts)

    categories_2 = np.concatenate([l[1] for l in layouts_2])
    counts = np.array(
        [np.count_nonzero(categories_2 == i) for i in range(n_categories)]
    )
    prob_2 = counts / np.sum(counts)
    return np.absolute(prob_1 - prob_2).sum()


def _compute_wasserstein_distance_bbox(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
) -> float:
    bboxes_1 = np.concatenate([l[0] for l in layouts_1]).T
    bboxes_2 = np.concatenate([l[0] for l in layouts_2]).T

    # simple 1-dimensional wasserstein for (cx, cy, w, h) independently
    N = 4
    ans = 0.0
    for i in range(N):
        ans += wasserstein_distance(bboxes_1[i], bboxes_2[i])
    ans /= N

    return ans


def compute_wasserstein_distance(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
    n_classes: int = 25,
) -> Dict[str, float]:
    w_class = _compute_wasserstein_distance_class(layouts_1, layouts_2, n_classes)
    w_bbox = _compute_wasserstein_distance_bbox(layouts_1, layouts_2)
    return {
        "wdist_class": w_class,
        "wdist_bbox": w_bbox,
    }


if __name__ == "__main__":
    layouts = [
        (
            np.array(
                [
                    [0.2, 0.2, 0.4, 0.4],
                ]
            ),
            np.zeros((1,)),
        )
    ]
    print(compute_average_iou(layouts))
