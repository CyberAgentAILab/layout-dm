from functools import partial

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from trainer.data.util import REL_SIZE_ALPHA, RelLoc, RelSize
from trainer.helpers.metric import compute_alignment, compute_overlap
from trainer.helpers.util import convert_xywh_to_ltrb


def beautify_alignment(bbox_flatten, data, threshold=0.004, **kwargs):
    bbox, mask = to_dense_batch(bbox_flatten, data.batch)
    bbox, mask = bbox[:, 1:], mask[:, 1:]

    if len(bbox_flatten.size()) == 3:
        bbox = bbox.transpose(1, 2)
        B, P, N, D = bbox.size()
        bbox = bbox.reshape(-1, N, D)
        mask = mask.unsqueeze(1).expand(-1, P, -1).reshape(-1, N)

    cost = compute_alignment(bbox, mask)
    cost = cost.masked_fill(cost.le(threshold), 0)

    if len(bbox_flatten.size()) == 3:
        cost = cost.view(B, P)

    return cost


def beautify_non_overlap(bbox_flatten, data, **kwargs):
    bbox, mask = to_dense_batch(bbox_flatten, data.batch)
    bbox, mask = bbox[:, 1:], mask[:, 1:]

    if len(bbox_flatten.size()) == 3:
        bbox = bbox.transpose(1, 2)
        B, P, N, D = bbox.size()
        bbox = bbox.reshape(-1, N, D)
        mask = mask.unsqueeze(1).expand(-1, P, -1).reshape(-1, N)

    cost = compute_overlap(bbox, mask)

    if len(bbox_flatten.size()) == 3:
        cost = cost.view(B, P)

    return cost


beautify = [beautify_alignment, beautify_non_overlap]


def less_equal(a, b):
    return torch.relu(a - b)


def less(a, b, eps=1e-8):
    return torch.relu(a - b + eps)


def _relation_size(rel_value, cost_func, bbox_flatten, data, canvas):
    cond = data.y[data.edge_index[0]].eq(0).eq(canvas)
    cond &= (data.edge_attr & 1 << rel_value).ne(0)

    if len(bbox_flatten.size()) == 3:
        cond = cond.unsqueeze(-1)
        a = bbox_flatten[:, :, 2] * bbox_flatten[:, :, 3]
    else:
        a = bbox_flatten[:, 2] * bbox_flatten[:, 3]

    ai, aj = a[data.edge_index[0]], a[data.edge_index[1]]

    cost = cost_func(ai, aj).masked_fill(~cond, 0)
    cost = to_dense_adj(data.edge_index, data.batch, cost)
    cost = cost.sum(dim=(1, 2))

    return cost


def relation_size_sm(bbox_flatten, data, canvas=False):
    def cost_func(a1, a2):
        # a2 <= a1_sm
        a1_sm = (1 - REL_SIZE_ALPHA) * a1
        return less_equal(a2, a1_sm)

    return _relation_size(RelSize.SMALLER, cost_func, bbox_flatten, data, canvas)


def relation_size_eq(bbox_flatten, data, canvas=False):
    def cost_func(a1, a2):
        # a1_sm < a2 and a2 < a1_lg
        a1_sm = (1 - REL_SIZE_ALPHA) * a1
        a1_lg = (1 + REL_SIZE_ALPHA) * a1
        return less(a1_sm, a2) + less(a2, a1_lg)

    return _relation_size(RelSize.EQUAL, cost_func, bbox_flatten, data, canvas)


def relation_size_lg(bbox_flatten, data, canvas=False):
    def cost_func(a1, a2):
        # a1_lg <= a2
        a1_lg = (1 + REL_SIZE_ALPHA) * a1
        return less_equal(a1_lg, a2)

    return _relation_size(RelSize.LARGER, cost_func, bbox_flatten, data, canvas)


def _relation_loc_canvas(rel_value, cost_func, bbox_flatten, data):
    cond = data.y[data.edge_index[0]].eq(0)
    cond &= (data.edge_attr & 1 << rel_value).ne(0)

    if len(bbox_flatten.size()) == 3:
        cond = cond.unsqueeze(-1)
        yc = bbox_flatten[:, :, 1]
    else:
        yc = bbox_flatten[:, 1]

    yc = yc[data.edge_index[1]]

    cost = cost_func(yc).masked_fill(~cond, 0)
    cost = to_dense_adj(data.edge_index, data.batch, cost)
    cost = cost.sum(dim=(1, 2))

    return cost


def relation_loc_canvas_t(bbox_flatten, data):
    def cost_func(yc):
        # yc <= y_sm
        y_sm = 1.0 / 3
        return less_equal(yc, y_sm)

    return _relation_loc_canvas(RelLoc.TOP, cost_func, bbox_flatten, data)


def relation_loc_canvas_c(bbox_flatten, data):
    def cost_func(yc):
        # y_sm < yc and yc < y_lg
        y_sm, y_lg = 1.0 / 3, 2.0 / 3
        return less(y_sm, yc) + less(yc, y_lg)

    return _relation_loc_canvas(RelLoc.CENTER, cost_func, bbox_flatten, data)


def relation_loc_canvas_b(bbox_flatten, data):
    def cost_func(yc):
        # y_lg <= yc
        y_lg = 2.0 / 3
        return less_equal(y_lg, yc)

    return _relation_loc_canvas(RelLoc.BOTTOM, cost_func, bbox_flatten, data)


def _relation_loc(rel_value, cost_func, bbox_flatten, data):
    cond = data.y[data.edge_index[0]].ne(0)
    cond &= (data.edge_attr & 1 << rel_value).ne(0)

    if len(bbox_flatten.size()) == 3:
        cond = cond.unsqueeze(-1)
        l, t, r, b = convert_xywh_to_ltrb(bbox_flatten.permute(2, 0, 1))
    else:
        l, t, r, b = convert_xywh_to_ltrb(bbox_flatten.t())

    li, lj = l[data.edge_index[0]], l[data.edge_index[1]]
    ti, tj = t[data.edge_index[0]], t[data.edge_index[1]]
    ri, rj = r[data.edge_index[0]], r[data.edge_index[1]]
    bi, bj = b[data.edge_index[0]], b[data.edge_index[1]]

    cost = cost_func(l1=li, t1=ti, r1=ri, b1=bi, l2=lj, t2=tj, r2=rj, b2=bj)

    if rel_value in [RelLoc.LEFT, RelLoc.RIGHT, RelLoc.CENTER]:
        # t1 < b2 and t2 < b1
        cost = cost + less(ti, bj) + less(tj, bi)

    cost = cost.masked_fill(~cond, 0)
    cost = to_dense_adj(data.edge_index, data.batch, cost)
    cost = cost.sum(dim=(1, 2))

    return cost


def relation_loc_t(bbox_flatten, data):
    def cost_func(b2, t1, **kwargs):
        # b2 <= t1
        return less_equal(b2, t1)

    return _relation_loc(RelLoc.TOP, cost_func, bbox_flatten, data)


def relation_loc_b(bbox_flatten, data):
    def cost_func(b1, t2, **kwargs):
        # b1 <= t2
        return less_equal(b1, t2)

    return _relation_loc(RelLoc.BOTTOM, cost_func, bbox_flatten, data)


def relation_loc_l(bbox_flatten, data):
    def cost_func(r2, l1, **kwargs):
        # r2 <= l1
        return less_equal(r2, l1)

    return _relation_loc(RelLoc.LEFT, cost_func, bbox_flatten, data)


def relation_loc_r(bbox_flatten, data):
    def cost_func(r1, l2, **kwargs):
        # r1 <= l2
        return less_equal(r1, l2)

    return _relation_loc(RelLoc.RIGHT, cost_func, bbox_flatten, data)


def relation_loc_c(bbox_flatten, data):
    def cost_func(l1, r2, l2, r1, **kwargs):
        # l1 < r2 and l2 < r1
        return less(l1, r2) + less(l2, r1)

    return _relation_loc(RelLoc.CENTER, cost_func, bbox_flatten, data)


relation = [
    partial(relation_size_sm, canvas=False),
    partial(relation_size_sm, canvas=True),
    partial(relation_size_eq, canvas=False),
    partial(relation_size_eq, canvas=True),
    partial(relation_size_lg, canvas=False),
    partial(relation_size_lg, canvas=True),
    relation_loc_canvas_t,
    relation_loc_canvas_c,
    relation_loc_canvas_b,
    relation_loc_t,
    relation_loc_b,
    relation_loc_l,
    relation_loc_r,
    relation_loc_c,
]
