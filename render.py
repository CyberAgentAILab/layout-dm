# Render generation results and conditional inputs if available
import argparse
import csv
import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange
from fsspec.core import url_to_fs
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from trainer.data.util import loader_to_list, sparse_to_dense
from trainer.helpers.metric import compute_alignment, compute_docsim, compute_overlap
from trainer.helpers.util import set_seed
from trainer.helpers.visualization import (
    save_image,
    save_label,
    save_label_with_size,
    save_relation,
)

CANVAS_SIZE = (120, 80)


def _repeat(inputs, n: int):
    outputs = []
    for x in inputs:
        for i in range(n):
            outputs.append(x)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="tmp/visualization")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--dump_num_samples", type=int, default=100)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="disable parallel computation for debugging",
    )
    args = parser.parse_args()
    set_seed(0)

    fs, _ = url_to_fs(args.result_dir)

    pkl_paths = [p for p in fs.ls(args.result_dir) if p.split(".")[-1] == "pkl"]
    pkl_paths = pkl_paths[:1]
    with fs.open(pkl_paths[0], "rb") as file_obj:
        meta = pickle.load(file_obj)
        train_cfg = meta["train_cfg"]
        test_cfg = meta["test_cfg"]

    train_cfg.data.num_workers = os.cpu_count()
    batch_size = args.eval_batch_size  # Note: arbitrary number is OK unless OOM

    kwargs = {
        "batch_size": batch_size,
        "num_workers": train_cfg.data.num_workers,
        "pin_memory": True,
        "shuffle": False,
    }

    split_main = "val" if test_cfg.get("is_validation", False) else "test"
    main_dataset = instantiate(train_cfg.dataset)(split=split_main, transform=None)
    if test_cfg.get("debug_num_samples", -1) > 0:
        num_samples = test_cfg.get("debug_num_samples")
    else:
        num_samples = args.dump_num_samples
    main_dataset = main_dataset[:num_samples]
    main_dataloader = DataLoader(main_dataset, **kwargs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name, ckpt_name, cond_name = args.result_dir.split("/")[-3:]
    pred_dir = Path(args.output_dir) / dataset_name / ckpt_name / cond_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.output_dir) / dataset_name / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    names_all = []
    for i, batch in enumerate(main_dataloader):
        bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
        names = [Path(n).with_suffix("").name for n in batch.attr["name"]]
        names_all.extend(names)
        for j, name in enumerate(names):
            out_path = gt_dir / f"{name}.png"
            if out_path.exists():
                continue
            save_image(
                bbox[j : j + 1],
                label[j : j + 1],
                mask[j : j + 1],
                main_dataset.colors,
                out_path,
                canvas_size=CANVAS_SIZE,
            )
    layouts_main = _repeat(loader_to_list(main_dataloader), test_cfg.num_run)
    names_all = _repeat(names_all, test_cfg.num_run)

    for t, pkl_path in enumerate(pkl_paths):
        headers = ["name", "alignment-LayoutGAN++", "overlap-LayoutGAN++", "docsim"]
        if test_cfg.cond in [
            "relation",
        ]:
            headers.append("violation")
        scores = []

        with fs.open(pkl_path, "rb") as file_obj:
            x = pickle.load(file_obj)

        N = num_samples * test_cfg.num_run
        generated = x["results"][:N]
        if "inputs" in x:
            inputs = x["inputs"][:N]
        if "relations" in x:
            relations = x["relations"][:N]
            relation_scores = x["relation_scores"][:N]

        for i, (b, l) in enumerate(tqdm(generated)):
            if test_cfg.cond in [
                "relation",
            ]:
                data = relations[i]
                id_ = data.attr["name"][0].split("/")[-1].split(".")[0]
            else:
                id_ = names_all[i]

            b, l = torch.from_numpy(b), torch.from_numpy(l)
            batched_b = rearrange(b, "s x -> 1 s x")
            batched_l = rearrange(l, "s -> 1 s")
            batched_m = torch.full(batched_l.size(), True)
            alignment = compute_alignment(batched_b, batched_m)
            overlap = compute_overlap(batched_b, batched_m)
            docsim = compute_docsim(
                [(b, l)],
                [list(torch.from_numpy(l) for l in layouts_main[i])],
            )

            score = [
                id_,
                alignment["alignment-LayoutGAN++"].item(),
                overlap["overlap-LayoutGAN++"].item(),
                docsim,
            ]

            is_first = i % test_cfg.num_run == 0

            pred_image_path = pred_dir / f"{id_}_{i % test_cfg.num_run}.png"
            if not pred_image_path.exists():
                save_image(
                    batched_b,
                    batched_l,
                    batched_m,
                    main_dataset.colors,
                    pred_image_path,
                    canvas_size=CANVAS_SIZE,
                )

            if test_cfg.cond in ["c", "cwh", "relation"] and is_first:
                pred_label_path = pred_dir / f"{id_}_label.png"
                if not pred_label_path.exists():
                    save_label(
                        l,
                        main_dataset.labels,
                        main_dataset.colors,
                        pred_label_path,
                        canvas_size=tuple([x * 3 for x in CANVAS_SIZE]),
                    )
            if test_cfg.cond == "cwh" and is_first:
                pred_label_w_size_path = pred_dir / f"{id_}_label_w_size.png"
                if not pred_label_w_size_path.exists():
                    save_label_with_size(
                        l,
                        main_dataset.labels,
                        main_dataset.colors,
                        pred_label_w_size_path,
                        canvas_size=tuple([x * 3 for x in CANVAS_SIZE]),
                    )

            if test_cfg.cond in ["partial", "refinement"] and is_first:
                input_b, input_l = inputs[i]
                batched_b = rearrange(torch.from_numpy(input_b), "s x -> 1 s x")
                batched_l = rearrange(torch.from_numpy(input_l), "s -> 1 s")
                batched_m = torch.full(batched_l.size(), True)
                input_path = pred_dir / f"{id_}_input.png"
                if not input_path.exists():
                    save_image(
                        batched_b,
                        batched_l,
                        batched_m,
                        main_dataset.colors,
                        input_path,
                        canvas_size=CANVAS_SIZE,
                    )

            if test_cfg.cond == "relation":
                if len(data.edge_index) == 0:
                    continue
                if is_first:
                    relation_path = pred_dir / f"{id_}_relation.png"
                    edge_attr = to_dense_adj(
                        data.edge_index, data.batch, data.edge_attr
                    )
                    save_relation(
                        data.y.cpu().numpy(),
                        edge_attr.cpu()[0],
                        main_dataset.labels,
                        main_dataset.colors,
                        relation_path,
                    )
                score.append(relation_scores[i])

            scores.append(score)

        with (pred_dir / "stats.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(scores)
        exit()


if __name__ == "__main__":
    main()
