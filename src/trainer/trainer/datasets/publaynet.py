import os

import torch
from fsspec.core import url_to_fs
from torch_geometric.data import Data
from tqdm import tqdm

from .base import BaseDataset


class PubLayNetDataset(BaseDataset):
    name = "publaynet"
    labels = [
        "text",
        "title",
        "list",
        "table",
        "figure",
    ]

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)

    def download(self):
        # super().download()
        pass

    def process(self):
        from pycocotools.coco import COCO

        fs, _ = url_to_fs(self.raw_dir)

        # if self.raw_dir.startswith("gs://"):
        #     raise NotImplementedError

        for split_publaynet in ["train", "val"]:
            data_list = []
            coco = COCO(
                os.path.join(self.raw_dir, "publaynet", f"{split_publaynet}.json")
            )
            for img_id in tqdm(sorted(coco.getImgIds())):
                ann_img = coco.loadImgs(img_id)
                W = float(ann_img[0]["width"])
                H = float(ann_img[0]["height"])
                name = ann_img[0]["file_name"]
                if H < W:
                    continue

                def is_valid(element):
                    x1, y1, width, height = element["bbox"]
                    x2, y2 = x1 + width, y1 + height
                    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                        return False

                    if x2 <= x1 or y2 <= y1:
                        return False

                    return True

                elements = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
                _elements = list(filter(is_valid, elements))
                filtered = len(elements) != len(_elements)
                elements = _elements

                N = len(elements)
                if N == 0 or self.max_seq_length < N:
                    continue

                boxes = []
                labels = []

                for element in elements:
                    # bbox
                    x1, y1, width, height = element["bbox"]
                    xc = x1 + width / 2.0
                    yc = y1 + height / 2.0
                    b = [xc / W, yc / H, width / W, height / H]
                    boxes.append(b)

                    # label
                    l = coco.cats[element["category_id"]]["name"]
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

            if split_publaynet == "train":
                train_list = data_list
            else:
                val_list = data_list

        # shuffle train with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(train_list), generator=generator)
        train_list = [train_list[i] for i in indices]

        # train_list -> train 95% / val 5%
        # val_list -> test 100%
        s = int(len(train_list) * 0.95)
        with fs.open(self.processed_paths[0], "wb") as file_obj:
            torch.save(self.collate(train_list[:s]), file_obj)
        with fs.open(self.processed_paths[1], "wb") as file_obj:
            torch.save(self.collate(train_list[s:]), file_obj)
        with fs.open(self.processed_paths[2], "wb") as file_obj:
            torch.save(self.collate(val_list), file_obj)
