import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from trainer.datasets import DATASETS
from trainer.helpers.bbox_tokenizer import KEY_MULT_DICT, BboxTokenizer

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_VOCABULARIES = ["pad", "bos", "eos", "mask"]
CHOICES = {
    "shared_bbox_vocab": ["xywh", "x-y-w-h"],
    "var_order": ["c-x-y-w-h", "c-w-h-x-y"],
    "bbox_quantization": ["linear", "kmeans"],
    # "bbox_quantization": ["linear", "kmeans", "percentile"],
}


def _pad_sequence(seq: LongTensor, max_seq_length: int, value: Any) -> LongTensor:
    S = seq.shape[1]
    new_shape = list(seq.shape)
    s = max_seq_length - S
    if s > 0:
        new_shape[1] = s
        pad = torch.full(new_shape, value, dtype=seq.dtype)
        new_seq = torch.cat([seq, pad], dim=1)
    else:
        new_seq = seq

    return new_seq


class LayoutTokenizer:
    """
    Tokenizer converts inputs into (dict of) a sequence
    This is a base class for all tokenizers
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ) -> None:
        self._data_cfg = data_cfg
        self._dataset_cfg = dataset_cfg
        self._sort_by = None

        name = dataset_cfg._target_.split(".")[-1]
        inv_dic = {str(v.__name__): k for (k, v) in DATASETS.items()}

        # validation
        self._var_order = data_cfg.get("var_order", "c-x-y-w-h")
        # assert self.var_order in ["c-x-y-w-h", "c-w-h-x-y", "c-xw-yh", "c-xywh"]
        assert self._var_order[0] == "c"
        assert all(token in SPECIAL_TOKEN_VOCABULARIES for token in self.special_tokens)
        if "mask" in self.special_tokens:
            assert self.special_tokens.index("mask") == self.N_sp_token - 1

        dataset_name = f"{inv_dic[name]}_max{dataset_cfg.max_seq_length}"
        self._bbox_tokenizer = BboxTokenizer(
            num_bin_bboxes=data_cfg.num_bin_bboxes,
            var_order=self._var_order,
            shared_bbox_vocab=data_cfg.get("shared_bbox_vocab", "xywh"),
            bbox_quantization=data_cfg.get("bbox_quantization", "linear"),
            dataset_name=dataset_name,
        )

        self._N_category = len(DATASETS[inv_dic[name]].labels)

        logger.info(
            f"N_total={self.N_total}, (N_label, N_bbox, N_sp_token)=({self.N_category},{self.N_bbox},{self.N_sp_token})"
        )

        self._special_token_name_to_id = {
            token: self.special_tokens.index(token) + self.N_category + self.N_bbox
            for token in self.special_tokens
        }
        self._special_token_id_to_name = {
            v: k for (k, v) in self._special_token_name_to_id.items()
        }

    def _pad_until(
        self, label: LongTensor, bbox: FloatTensor, mask: BoolTensor
    ) -> Tuple[LongTensor, FloatTensor, BoolTensor]:
        if self.pad_until_max:
            label = _pad_sequence(label, self.max_seq_length, 0)
            bbox = _pad_sequence(bbox, self.max_seq_length, 0)
            mask = _pad_sequence(mask, self.max_seq_length, False)
        return label, bbox, mask

    def _fix_padded_sequences(
        self, label: LongTensor, bbox: FloatTensor, mask: BoolTensor
    ) -> Tuple[LongTensor, FloatTensor]:
        pad_mask = ~mask
        if "pad" in self.special_tokens:
            pad_id = self.name_to_id("pad")
            label[pad_mask] = pad_id
            bbox[pad_mask] = pad_id
        return label, bbox

    def _filter_invalid_labels_and_bboxes(
        self, label: LongTensor, bbox: FloatTensor
    ) -> BoolTensor:
        # If a set of tokens for an element is corrupted, discard the element
        label_valid = (0 <= label) & (label < self.N_category)
        bbox_valid = (0 <= bbox) & (bbox < self.N_bbox)
        bbox_valid = torch.all(bbox_valid, dim=-1)
        invalid = torch.logical_not(label_valid & bbox_valid)
        return invalid

    def _filter_eos(self, label: LongTensor) -> BoolTensor:
        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            invalid = torch.cumsum(label == self.name_to_id("eos"), dim=1) > 0
        else:
            invalid = torch.full(label.size(), fill_value=False)
        return invalid

    @property
    def bbox_tokenizer(self) -> BboxTokenizer:
        return self._bbox_tokenizer

    @property
    def max_seq_length(self) -> int:
        return self._dataset_cfg.max_seq_length

    @property
    def max_token_length(self) -> int:
        return self.max_seq_length * self.N_var_per_element

    @property
    def N_bbox(self) -> int:
        return self.bbox_tokenizer.bbox_vocab_len

    @property
    def N_bbox_per_var(self) -> int:
        return self.bbox_tokenizer.num_bin_bboxes

    @property
    def N_category(self) -> int:
        return self._N_category

    @property
    def N_sp_token(self) -> int:
        return len(self.special_tokens)

    @property
    def N_total(self) -> int:
        return self.N_category + self.N_bbox + self.N_sp_token

    @property
    def N_var_per_element(self) -> int:
        return len(self.var_names)

    @property
    def special_tokens(self) -> List[str]:
        return self._data_cfg.special_tokens

    @property
    def pad_until_max(self) -> bool:
        return self._data_cfg.pad_until_max

    @property
    def var_names(self) -> List[str]:
        return self._var_order.split("-")

    @property
    def var_order(self) -> str:
        return self._var_order

    # functions below are for accesing special token properties
    def name_to_id(self, name: str) -> int:
        return self._special_token_name_to_id[name]

    def id_to_name(self, id_: int) -> str:
        return self._special_token_id_to_name[id_]

    @property
    def sort_by(self):
        return self._sort_by

    @sort_by.setter
    def sort_by(self, value: str = None):
        # return self._sort_by
        if value in ["None", "none"]:
            value = None
        if value is not None:
            assert value == "category_alphabetical"
            self._sort_by = value


class LayoutSequenceTokenizer(LayoutTokenizer):
    """
    Converts a layout into a sequence (c_1, x_1, y_1, w_1, h_1, c_2, ...)
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ) -> None:
        super().__init__(data_cfg, dataset_cfg)

    def encode(
        self,
        inputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        inputs has the following items
            mask: torch.BoolTensor of shape (B, S)
            label: torch.LongTensor of shape (B, S)
            bbox: torch.FloatTensor of shape (B, S, 4)
        """
        label = deepcopy(rearrange(inputs["label"], "b s -> b s 1"))
        bbox = deepcopy(self._bbox_tokenizer.encode(inputs["bbox"]))
        mask = deepcopy(inputs["mask"])

        label, bbox, mask = self._pad_until(label, bbox, mask)
        bbox += self.N_category  # add offset
        label, bbox = self._fix_padded_sequences(label, bbox, mask)

        B, S = label.size()[:2]
        C = self.N_var_per_element

        # sanity check
        seq_len = reduce(mask.int(), "b s -> b 1", reduction="sum")
        indices = rearrange(torch.arange(0, S), "s -> 1 s")
        assert torch.all(torch.logical_not(mask) == (seq_len <= indices)).item()

        if self.sort_by == "category_alphabetical":
            label, index = torch.sort(label, dim=1)
            bbox = torch.gather(bbox, dim=1, index=repeat(index, "b s 1 -> b s c", c=4))
            mask = torch.gather(mask, dim=1, index=rearrange(index, "b s 1 -> b s"))

        # make 1d sequence
        seq = torch.cat([label, bbox], axis=-1)
        seq = rearrange(seq, "b s x -> b (s x)")
        mask = repeat(mask, "b s -> b (s c)", c=C)

        # append BOS/EOS for autoregressive models
        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            indices = rearrange(torch.arange(0, S * C), "s -> 1 s")
            eos_mask = seq_len * C == indices
            seq[eos_mask] = self.name_to_id("eos")
            bos = torch.full((B, 1), self.name_to_id("bos"))
            seq = torch.cat([bos, seq], axis=-1)
            mask = torch.cat([torch.full((B, 1), fill_value=True), mask], axis=-1)

        return {"seq": seq.long(), "mask": mask}

    def decode(self, ids: LongTensor) -> Dict[str, Tensor]:
        ids = rearrange(ids, "b (s c) -> b s c", c=self.N_var_per_element)
        label, bbox = deepcopy(ids[..., 0]), deepcopy(ids[..., 1:])

        bbox -= self.N_category
        invalid = self._filter_eos(label)
        invalid = invalid | self._filter_invalid_labels_and_bboxes(label, bbox)

        bbox = self.bbox_tokenizer.decode(bbox)
        label[invalid] = 0
        bbox[invalid] = 0.0
        return {"bbox": bbox, "label": label, "mask": torch.logical_not(invalid)}

    @property
    def token_mask(self) -> BoolTensor:
        """
        Returns a bool tensor in shape (S, C), which is used to filter our invalid predictions
        E.g., predict high probs on x=1, while the location of token is for predicting a category
        """
        masks = self.bbox_tokenizer.token_mask
        last = BoolTensor(
            [False if x in ["bos", "mask"] else True for x in self.special_tokens]
        )

        masks["c"] = torch.cat(
            [
                torch.full((self.N_category,), True),
                torch.full((self.N_bbox,), False),
                last,
            ]
        )
        for key in self.var_names:
            if key == "c":
                continue
            masks[key] = torch.cat(
                [torch.full((self.N_category,), False), masks[key], last]
            )
        mask = torch.stack([masks[k] for k in self.var_names], dim=0)
        mask = repeat(mask, "x c -> (s x) c", s=self.max_seq_length)
        return mask

    def get_slice(self, name: str) -> slice:
        assert name == "special" or name in self.var_names
        if name == "special":
            start = self.N_category + self.N_bbox
            end = self.N_total
        elif name == "c":
            start, end = 0, self.N_category
        else:
            start = self.N_category
            if self.bbox_tokenizer.shared_bbox_vocab == "xywh":
                mult = 0
            elif self.bbox_tokenizer.shared_bbox_vocab == "x-y-w-h":
                mult = self.bbox_tokenizer.var_names.index(name)
            else:
                raise NotImplementedError
            start += mult * self.N_bbox_per_var
            end = start + self.N_bbox_per_var
        return slice(start, end)


class LayoutDictTokenizer(LayoutTokenizer):
    """
    Converts a layout into a dict of a sequence
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ) -> None:
        super().__init__(data_cfg, dataset_cfg)
        # assert all(token in ["bos", "eos"] for token in data_cfg.special_tokens)
        assert data_cfg.var_order == "c-x-y-w-h"
        assert data_cfg.shared_bbox_vocab == "xywh"

    def encode(
        self,
        inputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        inputs has the following items
            mask: torch.BoolTensor of shape (B, S)
            label: torch.LongTensor of shape (B, S)
            bbox: torch.FloatTensor of shape (B, S, 4)
        """
        label = deepcopy(inputs["label"])  # (B, S)
        bbox = deepcopy(self.bbox_tokenizer.encode(inputs["bbox"]))  # (B, S, 4)
        mask = deepcopy(inputs["mask"])  # (B, S)

        label, bbox, mask = self._pad_until(label, bbox, mask)
        bbox += self.N_category  # add offset
        label, bbox = self._fix_padded_sequences(label, bbox, mask)

        B, S = label.size()[:2]
        C = self.N_var_per_element

        seq_len = reduce(mask.int(), "b s -> b 1", reduction="sum")
        indices = rearrange(torch.arange(0, mask.size(1)), "s -> 1 s")
        assert torch.all(torch.logical_not(mask) == (seq_len <= indices)).item()

        outputs = {"label": label, "bbox": bbox, "mask": mask}
        if "eos" in self.special_tokens and "bos" in self.special_tokens:
            # Add BOS
            B, S = mask.size()
            bos = {}
            bos["label"] = torch.full((B, 1), fill_value=self.name_to_id("bos")).to(
                label
            )
            bos["bbox"] = torch.full((B, 1, 4), fill_value=0).to(bbox)
            bos["mask"] = torch.full((B, 1), fill_value=True).to(mask)
            for key in outputs:
                outputs[key] = torch.cat([bos[key], outputs[key]], dim=1)

            # Add EOS
            S = outputs["mask"].size(1)
            indices = torch.arange(0, S).view(1, S)
            eos_mask = reduce(outputs["mask"], "b s -> b 1", reduction="sum") == indices
            outputs["label"][eos_mask] = self.name_to_id("eos")
            outputs["mask"] = eos_mask | outputs["mask"]

        return outputs

    def decode(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        label = deepcopy(inputs["label"])
        bbox = deepcopy(inputs["bbox"])
        if "mask" in inputs:
            mask = deepcopy(inputs["mask"])
            invalid = ~mask
        else:
            invalid = torch.full(inputs["label"].size(), False)

        bbox -= self.N_category

        invalid = invalid | self._filter_eos(label)
        invalid = invalid | self._filter_invalid_labels_and_bboxes(label, bbox)

        bbox = self.bbox_tokenizer.decode(bbox)
        label[invalid] = 0
        bbox[invalid] = 0.0

        return {"bbox": bbox, "label": label, "mask": ~invalid}


def _bucketize(
    inputs: LongTensor,
    from_ids: LongTensor,
    to_ids: LongTensor,
) -> LongTensor:
    """
    Map a set of ids to a different set of ids
    """
    assert from_ids.size() == to_ids.size()
    assert set(inputs.unique().tolist()) <= set(from_ids.tolist())
    index = torch.bucketize(inputs.ravel(), from_ids)
    remapped = to_ids[index].reshape(inputs.shape).to(inputs)
    return remapped


class Converter:
    """
    Convert full matrix (handling all vocabularies for cxywh) and partial matrix (handling each)
    f and p is short for full and partial, respectively.
    """

    def __init__(
        self,
        tokenizer: LayoutSequenceTokenizer,
    ):
        self.tokenizer = tokenizer
        self.C = len(tokenizer.var_names)
        c_pad_id = tokenizer.N_category + tokenizer.N_bbox
        c_mask_id = c_pad_id + 1
        self.mapping = {}
        self.mapping["c"] = {
            "partial": list(range(tokenizer.N_category + 2)),
            "full": list(range(tokenizer.N_category)) + [c_pad_id, c_mask_id],
        }

        mult_dict = KEY_MULT_DICT[tokenizer.bbox_tokenizer.shared_bbox_vocab]

        # for at_once mathods
        _offset = {}
        _offset["normal_f_to_p"] = [
            0,
        ]
        _offset["special_f_to_p"] = [-tokenizer.N_bbox] + [
            -c_pad_id + self.tokenizer.N_bbox_per_var for _ in range(self.C - 1)
        ]
        _offset["boundary_f_to_p"] = [c_pad_id for _ in range(self.C)]
        _offset["normal_p_to_f"] = [
            0,
        ]
        _offset["special_p_to_f"] = [
            tokenizer.N_bbox,
        ] + [c_pad_id - self.tokenizer.N_bbox_per_var for _ in range(self.C - 1)]
        _offset["boundary_p_to_f"] = [
            tokenizer.N_category,
        ] + [self.tokenizer.N_bbox_per_var for _ in range(self.C - 1)]

        for key in tokenizer.var_names:
            if key == "c":
                continue
            num_bin = tokenizer.N_bbox_per_var
            special_ids = [tokenizer.name_to_id("pad"), tokenizer.name_to_id("mask")]

            start = tokenizer.N_category
            if key in mult_dict:
                start += mult_dict[key] * num_bin
            self.mapping[key] = {
                "partial": list(range(num_bin + 2)),
                "full": list(range(start, start + num_bin)) + special_ids,
            }
            _offset["normal_f_to_p"].append(-start)
            _offset["normal_p_to_f"].append(start)

        for k, v in self.mapping.items():
            self.mapping[k] = {x: torch.LongTensor(y) for (x, y) in v.items()}
        for k, v in _offset.items():
            _offset[k] = torch.LongTensor(v)

        # pre-allocate to avoid calling repeat in every call
        self.batched_mapping, self.batched_offset = {}, {}
        B, S = 512, self.tokenizer.max_seq_length
        for k, v in self.mapping.items():
            self.batched_mapping[k] = {
                x: repeat(y, "c -> b c s", b=B, s=S) for (x, y) in v.items()
            }
        for k, v in _offset.items():
            self.batched_offset[k] = repeat(
                v, "x -> b s x", b=B, s=self.tokenizer.max_seq_length
            )

    def __call__(self):
        raise NotImplementedError

    def p_to_f_id(self, inputs: LongTensor, key: str) -> LongTensor:
        outputs = _bucketize(
            inputs=inputs,
            from_ids=self.mapping[key]["partial"],
            to_ids=self.mapping[key]["full"],
        )
        return outputs

    def p_to_f_id_all(self, ids_p: LongTensor) -> LongTensor:
        """
        p_to_f_id for all the layout tokens at once for efficiency.
        Note: the shape of ids is (B, S, C), where C = len(tokenizer.N_var_per_element),
        """
        B, S, C = ids_p.size()
        assert C == self.C
        ids_normal_f = ids_p + self.batched_offset["normal_p_to_f"][:B, :S]
        ids_special_f = ids_p + self.batched_offset["special_p_to_f"][:B, :S]
        ids_f = torch.where(
            ids_p < self.batched_offset["boundary_p_to_f"][:B, :S],
            ids_normal_f,
            ids_special_f,
        )
        return ids_f

    def f_to_p_id(self, inputs: LongTensor, key: str) -> LongTensor:
        outputs = _bucketize(
            inputs=inputs,
            from_ids=self.mapping[key]["full"],
            to_ids=self.mapping[key]["partial"],
        )
        return outputs

    def f_to_p_id_all(self, ids_f: LongTensor) -> LongTensor:
        """
        f_to_p_id for all the layout tokens at once for efficiency.
        Note: the shape of ids is (B, S, C), where C = len(tokenizer.N_var_per_element),
        """
        B, S, C = ids_f.size()
        assert C == self.C
        assert B <= 512
        ids_normal_p = ids_f + self.batched_offset["normal_f_to_p"][:B, :S]
        ids_special_p = ids_f + self.batched_offset["special_f_to_p"][:B, :S]
        ids_p = torch.where(
            ids_f < self.batched_offset["boundary_f_to_p"][:B, :S],
            ids_normal_p,
            ids_special_p,
        )
        return ids_p

    def p_to_f_log(self, inputs: FloatTensor, key: str) -> FloatTensor:
        B, _, S = inputs.size()
        assert B <= 512
        shape = (B, self.tokenizer.N_total, S)
        outputs = torch.full(shape, math.log(1e-30)).to(inputs)
        index = self.batched_mapping[key]["full"][:B, :, :S]
        outputs.scatter_(dim=1, index=index, src=inputs)
        return outputs

    def f_to_p_log(
        self,
        inputs: FloatTensor,
        key: str,
    ) -> FloatTensor:
        B, _, S = inputs.size()
        index = self.batched_mapping[key]["full"][:B, :, :S]
        outputs = torch.gather(inputs, dim=1, index=index)
        return outputs

    def to(self, device=torch.device) -> None:
        for k, v in self.mapping.items():
            self.mapping[k] = {x: y.to(device) for (x, y) in v.items()}
        for k, v in self.batched_mapping.items():
            self.batched_mapping[k] = {x: y.to(device) for (x, y) in v.items()}
        for k, v in self.batched_offset.items():
            self.batched_offset[k] = v.to(device)

    def get_device(self) -> torch.device:
        return self.mapping["c"]["full"].device
