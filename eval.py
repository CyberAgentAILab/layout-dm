import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from fsspec.core import url_to_fs
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from trainer.data.util import loader_to_list, sparse_to_dense
from trainer.fid.model import load_fidnet_v3
from trainer.global_configs import FID_WEIGHT_DIR
from trainer.helpers.metric import (
    Layout,
    compute_alignment,
    compute_average_iou,
    compute_docsim,
    compute_generative_model_scores,
    compute_maximum_iou,
    compute_overlap,
)
from trainer.helpers.util import set_seed


def preprocess(layouts: List[Layout], max_len: int, device: torch.device):
    layout = defaultdict(list)
    for (b, l) in layouts:
        pad_len = max_len - l.shape[0]
        bbox = torch.tensor(
            np.concatenate([b, np.full((pad_len, 4), 0.0)], axis=0),
            dtype=torch.float,
        )
        layout["bbox"].append(bbox)
        label = torch.tensor(
            np.concatenate([l, np.full((pad_len,), 0)], axis=0),
            dtype=torch.long,
        )
        layout["label"].append(label)
        mask = torch.tensor(
            [True for _ in range(l.shape[0])] + [False for _ in range(pad_len)]
        )
        layout["mask"].append(mask)
    bbox = torch.stack(layout["bbox"], dim=0).to(device)
    label = torch.stack(layout["label"], dim=0).to(device)
    mask = torch.stack(layout["mask"], dim=0).to(device)
    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def print_scores(scores: Dict, test_cfg: argparse.Namespace, train_cfg: DictConfig):
    scores = {k: scores[k] for k in sorted(scores)}
    job_name = train_cfg.job_dir.split("/")[-1]
    model_name = train_cfg.model._target_.split(".")[-1]
    cond = test_cfg.cond

    if "num_timesteps" in test_cfg:
        step = test_cfg.num_timesteps
    else:
        step = train_cfg.sampling.get("num_timesteps", None)

    option = ""
    header = ["job_name", "model_name", "cond", "step", "option"]
    data = [job_name, model_name, cond, step, option]

    tex = ""
    for k, v in scores.items():
        # if k == "Alignment" or k == "Overlap" or "Violation" in k:
        #     v = [_v * 100 for _v in v]
        mean, std = np.mean(v), np.std(v)
        stdp = std * 100.0 / mean
        print(f"\t{k}: {mean:.4f} ({stdp:.4f}%)")
        tex += f"& {mean:.4f}\\std{{{stdp:.1f}}}\% "

        header.extend([f"{k}-mean", f"{k}-std"])
        data.extend([mean, std])

    print(tex + "\\\\")

    print(",".join(header))
    print(",".join([str(d) for d in data]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str, default="tmp/results")
    parser.add_argument(
        "--compute_real",
        action="store_true",
        help="compute some metric between validation and test subset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="number of samples used for evaluating unconditional generation",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    set_seed(0)

    fs, _ = url_to_fs(args.result_dir)
    pkl_paths = [p for p in fs.ls(args.result_dir) if p.split(".")[-1] == "pkl"]
    with fs.open(pkl_paths[0], "rb") as file_obj:
        meta = pickle.load(file_obj)
        train_cfg, test_cfg = meta["train_cfg"], meta["test_cfg"]
        assert test_cfg.num_run == 1

    train_cfg.data.num_workers = os.cpu_count()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": train_cfg.data.num_workers,
        "pin_memory": True,
        "shuffle": False,
    }

    if test_cfg.get("is_validation", False):
        split_main, split_sub = "val", "test"
    else:
        split_main, split_sub = "test", "val"

    main_dataset = instantiate(train_cfg.dataset)(split=split_main, transform=None)
    if test_cfg.get("debug_num_samples", -1) > 0:
        main_dataset = main_dataset[: test_cfg.debug_num_samples]
    main_dataloader = DataLoader(main_dataset, **kwargs)
    layouts_main = loader_to_list(main_dataloader)

    if args.compute_real:
        sub_dataset = instantiate(train_cfg.dataset)(split=split_sub, transform=None)
        if test_cfg.cond == "unconditional":
            sub_dataset = sub_dataset[: args.num_samples]
        sub_dataloader = DataLoader(sub_dataset, **kwargs)
        layouts_sub = loader_to_list(sub_dataloader)

    num_classes = len(main_dataset.labels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fid_model = load_fidnet_v3(main_dataset, FID_WEIGHT_DIR, device)

    scores_all = defaultdict(list)
    feats_1 = []
    batch_metrics = defaultdict(float)
    for i, batch in enumerate(main_dataloader):
        bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
        with torch.set_grad_enabled(False):
            feat = fid_model.extract_features(bbox, label, padding_mask)
        feats_1.append(feat.cpu())
        # save_image(bbox, label, mask, main_dataset.colors, f"dummy.png")

        if args.compute_real:
            for k, v in compute_alignment(bbox.cpu(), mask.cpu()).items():
                batch_metrics[k] += v.sum().item()
            for k, v in compute_overlap(bbox.cpu(), mask.cpu()).items():
                batch_metrics[k] += v.sum().item()

    if args.compute_real:
        scores_real = defaultdict(list)
        for k, v in batch_metrics.items():
            scores_real.update({k: v / len(main_dataset)})

    # compute metrics between real val and test dataset
    if args.compute_real:
        feats_1_another = []
        for batch in sub_dataloader:
            bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            feats_1_another.append(feat.cpu())

        scores_real.update(compute_generative_model_scores(feats_1, feats_1_another))
        scores_real.update(compute_average_iou(layouts_sub))
        if test_cfg.cond != "unconditional":
            scores_real["maximum_iou"] = compute_maximum_iou(layouts_main, layouts_sub)
            scores_real["DocSim"] = compute_docsim(layouts_main, layouts_main)

        # regard as the result of single run
        scores_real = {k: [v] for (k, v) in scores_real.items()}
        print()
        print("\nReal data:")
        print_scores(scores_real, test_cfg, train_cfg)

    # compute scores for each run
    for pkl_path in pkl_paths:
        feats_2 = []
        batch_metrics = defaultdict(float)

        with fs.open(pkl_path, "rb") as file_obj:
            x = pickle.load(file_obj)
        generated = x["results"]

        for i in range(0, len(generated), args.batch_size):
            i_end = min(i + args.batch_size, len(generated))
            batch = generated[i:i_end]
            max_len = max(len(g[-1]) for g in batch)

            bbox, label, padding_mask, mask = preprocess(batch, max_len, device)
            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            feats_2.append(feat.cpu())

            for k, v in compute_alignment(bbox, mask).items():
                batch_metrics[k] += v.sum().item()
            for k, v in compute_overlap(bbox, mask).items():
                batch_metrics[k] += v.sum().item()

        scores = {}
        for k, v in batch_metrics.items():
            scores[k] = v / len(generated)
        scores.update(compute_average_iou(generated))
        scores.update(compute_generative_model_scores(feats_1, feats_2))
        if test_cfg.cond != "unconditional":
            scores["maximum_iou"] = compute_maximum_iou(layouts_main, generated)
            scores["DocSim"] = compute_docsim(layouts_main, generated)

        for k, v in scores.items():
            scores_all[k].append(v)

    print_scores(scores_all, test_cfg, train_cfg)
    print()


if __name__ == "__main__":
    main()
