import copy
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from fsspec.core import url_to_fs
from PIL import Image, ImageDraw
from torch_geometric.data import Data
from tqdm import tqdm
from trainer.data.util import sparse_to_dense
from trainer.helpers.util import convert_xywh_to_ltrb

from .base import BaseDataset

_rico5_labels = [
    "Text",
    "Text Button",
    "Toolbar",
    "Image",
    "Icon",
]

_rico13_labels = [
    "Toolbar",
    "Image",
    "Text",
    "Icon",
    "Text Button",
    "Input",
    "List Item",
    "Advertisement",
    "Pager Indicator",
    "Web View",
    "Background Image",
    "Drawer",
    "Modal",
]

_rico25_labels = [
    "Text",
    "Image",
    "Icon",
    "Text Button",
    "List Item",
    "Input",
    "Background Image",
    "Card",
    "Web View",
    "Radio Button",
    "Drawer",
    "Checkbox",
    "Advertisement",
    "Modal",
    "Pager Indicator",
    "Slider",
    "On/Off Switch",
    "Button Bar",
    "Toolbar",
    "Number Stepper",
    "Multi-Tab",
    "Date Picker",
    "Map View",
    "Video",
    "Bottom Navigation",
]


def append_child(element, elements):
    if "children" in element.keys():
        for child in element["children"]:
            elements.append(child)
            elements = append_child(child, elements)
    return elements


class _RicoDataset(BaseDataset):
    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)

    def process(self):
        data_list = []
        raw_file = os.path.join(
            self.raw_dir, "rico_dataset_v0.1_semantic_annotations.zip"
        )
        fs, _ = url_to_fs(self.raw_dir)
        with fs.open(raw_file, "rb") as f, ZipFile(f) as z:
            names = sorted([n for n in z.namelist() if n.endswith(".json")])
            for name in tqdm(names):
                ann = json.loads(z.open(name).read())

                B = ann["bounds"]
                W, H = float(B[2]), float(B[3])
                if B[0] != 0 or B[1] != 0 or H < W:
                    continue

                def is_valid(element):
                    if element["componentLabel"] not in set(self.labels):
                        print(element["componentLabel"])
                        return False

                    x1, y1, x2, y2 = element["bounds"]
                    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                        return False

                    if x2 <= x1 or y2 <= y1:
                        return False

                    return True

                elements = append_child(ann, [])
                _elements = list(filter(is_valid, elements))
                filtered = len(elements) != len(_elements)
                elements = _elements
                N = len(elements)
                if N == 0 or self.max_seq_length < N:
                    continue

                # only for debugging slice-based preprocessing
                # elements = append_child(ann, [])
                # filtered = False
                # if len(elements) == 0:
                #     continue
                # elements = elements[:self.max_seq_length]

                boxes = []
                labels = []

                for element in elements:
                    # bbox
                    x1, y1, x2, y2 = element["bounds"]
                    xc = (x1 + x2) / 2.0
                    yc = (y1 + y2) / 2.0
                    width = x2 - x1
                    height = y2 - y1
                    b = [xc / W, yc / H, width / W, height / H]
                    boxes.append(b)

                    # label
                    l = element["componentLabel"]
                    labels.append(self.label2index[l])

                boxes = torch.tensor(boxes, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)

                data = Data(x=boxes, y=labels)
                data.attr = {
                    "name": name,
                    "width": W,
                    "height": H,
                    "filtered": filtered,
                    "has_canvas_element": False,
                    "NoiseAdded": False,
                }
                data_list.append(data)

        # shuffle with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(data_list), generator=generator)
        data_list = [data_list[i] for i in indices]

        # train 85% / val 5% / test 10%
        N = len(data_list)
        s = [int(N * 0.85), int(N * 0.90)]

        with fs.open(self.processed_paths[0], "wb") as file_obj:
            torch.save(self.collate(data_list[: s[0]]), file_obj)
        with fs.open(self.processed_paths[1], "wb") as file_obj:
            torch.save(self.collate(data_list[s[0] : s[1]]), file_obj)
        with fs.open(self.processed_paths[2], "wb") as file_obj:
            torch.save(self.collate(data_list[s[1] :]), file_obj)

    def download(self):
        pass

    def get_original_resource(self, batch) -> Image:
        assert not self.raw_dir.startswith("gs://")
        bbox, _, _, _ = sparse_to_dense(batch)

        img_bg, img_original, cropped_patches = [], [], []
        names = batch.attr["name"]
        if isinstance(names, str):
            names = [names]

        for i, name in enumerate(names):
            name = Path(name).name.replace(".json", ".jpg")
            img = Image.open(Path(self.raw_dir) / "combined" / name)
            img_original.append(copy.deepcopy(img))

            W, H = img.size
            ltrb = convert_xywh_to_ltrb(bbox[i].T.numpy())
            left, right = (ltrb[0] * W).astype(np.uint32), (ltrb[2] * W).astype(
                np.uint32
            )
            top, bottom = (ltrb[1] * H).astype(np.uint32), (ltrb[3] * H).astype(
                np.uint32
            )
            draw = ImageDraw.Draw(img)
            patches = []
            for (l, r, t, b) in zip(left, right, top, bottom):
                patches.append(img.crop((l, t, r, b)))
                # draw.rectangle([(l, t), (r, b)], fill=(255, 0, 0))
                draw.rectangle([(l, t), (r, b)], fill=(255, 255, 255))
            img_bg.append(img)
            cropped_patches.append(patches)
            # if len(patches) < S:
            #     for i in range(S - len(patches)):
            #         patches.append(Image.new("RGB", (0, 0)))

        return {
            "img_bg": img_bg,
            "img_original": img_original,
            "cropped_patches": cropped_patches,
        }

        # read from uncompressed data (the last line takes infinite time, so not used now..)
        # raw_file = os.path.join(self.raw_dir, "unique_uis.tar.gz")
        # with tarfile.open(raw_file) as f:
        #     # return gzip.GzipFile(fileobj=f.extractfile(f"combined/{name}")).read()
        #     return gzip.GzipFile(fileobj=f.extractfile(f"combined/hoge")).read()


class Rico5Dataset(_RicoDataset):
    name = "rico5"
    labels = _rico5_labels

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)


# Constrained Graphic Layout Generation via Latent Optimization (ACMMM2021)
class Rico13Dataset(_RicoDataset):
    name = "rico13"
    labels = _rico13_labels

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)


class Rico25Dataset(_RicoDataset):
    name = "rico25"
    labels = _rico25_labels

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)
