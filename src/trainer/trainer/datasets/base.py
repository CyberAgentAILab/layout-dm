import os

import fsspec
import seaborn as sns
import torch
from fsspec.core import url_to_fs

from .dataset import InMemoryDataset


class BaseDataset(InMemoryDataset):
    name = None
    labels = []
    _label2index = None
    _index2label = None
    _colors = None

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        assert split in ["train", "val", "test"]
        name = f"{self.name}-max{max_seq_length}"
        self.max_seq_length = max_seq_length
        super().__init__(os.path.join(dir, name), transform)
        idx = self.processed_file_names.index("{}.pt".format(split))

        with fsspec.open(self.processed_paths[idx], "rb") as file_obj:
            self.data, self.slices = torch.load(file_obj)

    @property
    def label2index(self):
        if self._label2index is None:
            self._label2index = dict()
            for idx, label in enumerate(self.labels):
                self._label2index[label] = idx
        return self._label2index

    @property
    def index2label(self):
        if self._index2label is None:
            self._index2label = dict()
            for idx, label in enumerate(self.labels):
                self._index2label[idx] = label
        return self._index2label

    @property
    def colors(self):
        if self._colors is None:
            n_colors = self.num_classes
            colors = sns.color_palette("husl", n_colors=n_colors)
            self._colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
        return self._colors

    @property
    def raw_file_names(self):
        fs, _ = url_to_fs(self.raw_dir)
        if not fs.exists(self.raw_dir):
            return []
        file_names = [f.split("/")[-1] for f in fs.ls(self.raw_dir)]
        return file_names

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        raise FileNotFoundError("See dataset/README.md")

    def process(self):
        raise NotImplementedError

    def get_original_images(self):
        raise NotImplementedError
