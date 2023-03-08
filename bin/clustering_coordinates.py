import argparse
import logging
import pickle
import time
from pathlib import Path

import omegaconf
import torch
from fsspec.core import url_to_fs
from hydra.utils import instantiate
from sklearn.cluster import KMeans
from trainer.global_configs import DATASET_DIR
from trainer.helpers.clustering import Percentile

logger = logging.getLogger(__name__)

KEYS = ["x", "y", "w", "h"]

parser = argparse.ArgumentParser()
parser.add_argument("dataset_yaml", type=str)
parser.add_argument("algorithm", type=str, choices=["kmeans", "percentile"])
parser.add_argument("--result_dir", type=str, default="tmp/clustering_weights")
parser.add_argument("--random_state", type=int, default=0)
parser.add_argument(
    "--max_bbox_num",
    type=int,
    default=int(1e5),
    help="filter number of bboxes to avoid too much time consumption in kmeans",
)

args = parser.parse_args()
fs, _ = url_to_fs(args.dataset_yaml)
n_clusters_list = [2**i for i in range(1, 9)]

dataset_cfg = omegaconf.OmegaConf.load(args.dataset_yaml)
dataset_cfg["dir"] = DATASET_DIR
dataset = instantiate(dataset_cfg)(split="train", transform=None)
bboxes = torch.cat([e.x for e in dataset], axis=0)

models = {}
name = Path(args.dataset_yaml).stem
weight_path = f"{args.result_dir}/{name}_max{dataset_cfg.max_seq_length}_{args.algorithm}_train_clusters.pkl"

if bboxes.size(0) > args.max_bbox_num and args.algorithm == "kmeans":
    text = f"{bboxes.size(0)} -> {args.max_bbox_num}"
    logger.warning(
        f"Subsampling bboxes because there are too many for kmeans: ({text})"
    )
    generator = torch.Generator().manual_seed(args.random_state)
    indices = torch.randperm(bboxes.size(0), generator=generator)
    bboxes = bboxes[indices[: args.max_bbox_num]]

for n_clusters in n_clusters_list:
    start_time = time.time()
    if args.algorithm == "kmeans":
        kwargs = {"n_clusters": n_clusters, "random_state": args.random_state}
        # one variable
        for i, key in enumerate(KEYS):
            key = f"{key}-{n_clusters}"
            models[key] = KMeans(**kwargs).fit(bboxes[..., i : i + 1].numpy())
    elif args.algorithm == "percentile":
        kwargs = {"n_clusters": n_clusters}
        for i, key in enumerate(KEYS):
            key = f"{key}-{n_clusters}"
            models[key] = Percentile(**kwargs).fit(bboxes[..., i : i + 1].numpy())
    print(
        f"{name} ({args.algorithm} {n_clusters} clusters): {time.time() - start_time}s"
    )

with fs.open(weight_path, "wb") as f:
    pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
