import copy
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Dict

import hydra
import numpy as np
import torch
import torchvision.transforms as T
from einops import repeat
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from trainer.crossplatform_util import filter_args_for_ai_platform
from trainer.data.util import (
    AddCanvasElement,
    AddRelationConstraints,
    split_num_samples,
)
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.metric import compute_violation
from trainer.helpers.sampling import SAMPLING_CONFIG_DICT
from trainer.helpers.task import get_cond
from trainer.helpers.util import set_seed
from trainer.helpers.visualization import save_image
from trainer.models.common.util import load_model

from .global_configs import DATASET_DIR
from .hydra_configs import TestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _filter_invalid(layouts: Dict[str, Tensor]):
    outputs = []
    for b in range(layouts["bbox"].size(0)):
        bbox = layouts["bbox"][b].numpy()
        label = layouts["label"][b].numpy()
        mask = layouts["mask"][b].numpy()
        outputs.append((bbox[mask], label[mask]))
    return outputs


# instantiate a hydra config for test
cs = ConfigStore.instance()
cs.store(name="test_config", node=TestConfig)


@hydra.main(version_base="1.2", config_name="test_config")
def main(test_cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_per_input = test_cfg.num_run > 1

    fs, _ = url_to_fs(test_cfg.job_dir)
    if not fs.exists(test_cfg.job_dir):
        raise FileNotFoundError

    config_path = os.path.join(test_cfg.job_dir, "config.yaml")
    if fs.exists(config_path):
        with fs.open(config_path, "rb") as file_obj:
            train_cfg = OmegaConf.load(file_obj)
        ckpt_dirs = [test_cfg.job_dir]
    else:
        # multi-seed experiment
        # assume seed is 0, 1, 2, ...
        ckpt_dirs = []
        seed = 0
        while True:
            tmp_job_dir = os.path.join(test_cfg.job_dir, str(seed))
            config_path = os.path.join(tmp_job_dir, "config.yaml")

            if fs.exists(config_path):
                if seed == 0:
                    with fs.open(config_path, "rb") as file_obj:
                        train_cfg = OmegaConf.load(file_obj)
                ckpt_dirs.append(tmp_job_dir)
            else:
                break

            seed += 1

    if test_cfg.debug:
        ckpt_dirs = [ckpt_dirs[0]]

    dataset_cfg = train_cfg.dataset
    if test_cfg.get("dataset_dir", None):
        dataset_cfg.dir = test_cfg.dataset_dir
    else:
        dataset_cfg.dir = DATASET_DIR

    data_cfg = train_cfg.data
    data_cfg.pad_until_max = True

    # if test_cfg.cond not in ["refinement", "unconditional"]:
    #     assert train_cfg.data.transforms == ["RandomOrder"]

    sampling_cfg = OmegaConf.structured(SAMPLING_CONFIG_DICT[test_cfg.sampling])
    OmegaConf.set_struct(sampling_cfg, False)
    if "temperature" in test_cfg:
        sampling_cfg.temperature = test_cfg.temperature
    if "top_p" in test_cfg and sampling_cfg.name == "top_p":
        sampling_cfg.top_p = test_cfg.top_p

    tokenizer = LayoutSequenceTokenizer(data_cfg=data_cfg, dataset_cfg=dataset_cfg)
    model = instantiate(train_cfg.model)(
        backbone_cfg=train_cfg.backbone, tokenizer=tokenizer
    )
    model = model.to(device)

    sampling_cfg = model.aggregate_sampling_settings(sampling_cfg, test_cfg)
    logger.warning(f"Test config: {test_cfg}")
    logger.warning(f"Sampling config: {sampling_cfg}")

    key = "_".join([f"{k}_{v}" for (k, v) in sampling_cfg.items()])
    if test_cfg.is_validation:
        key += "_validation"
    if test_cfg.debug:
        key += "_debug"
    if test_cfg.debug_num_samples > 0:
        key += f"_only_{test_cfg.debug_num_samples}_samples"

    if multi_per_input:
        assert test_cfg.cond
        test_cfg.max_batch_size = 1  # load single sample and generate multiple results
        key += f"_{test_cfg.num_run}samples_per_input"

    result_dir = os.path.join(test_cfg.result_dir, f"{test_cfg.cond}_{key}")
    if not fs.exists(result_dir):
        fs.mkdir(result_dir)
    logger.warning(f"Results saved to {result_dir}")

    scores = defaultdict(list)
    for seed_no, ckpt_dir in enumerate(ckpt_dirs):
        set_seed(seed_no)
        batch_metrics = defaultdict(float)
        model = load_model(
            model=model,
            ckpt_dir=ckpt_dir,
            device=device,
            best_or_final="best",
        )
        model.eval()

        if test_cfg.cond == "relation":
            test_transform = T.Compose(
                [
                    AddCanvasElement(),
                    AddRelationConstraints(seed=seed_no, edge_ratio=0.1),
                ]
            )
        else:
            test_transform = None

        split = "val" if test_cfg.is_validation else "test"
        dataset = instantiate(dataset_cfg)(split=split, transform=test_transform)
        if test_cfg.debug_num_samples > 0:
            dataset = dataset[: test_cfg.debug_num_samples]

        t_total = 0.0
        N_total = 0
        inputs, relations, relation_scores, results = [], [], [], []
        if test_cfg.cond == "unconditional":
            dataloader = split_num_samples(
                test_cfg.num_uncond_samples, test_cfg.max_batch_size
            )
        else:
            dataloader = DataLoader(
                dataset, batch_size=test_cfg.max_batch_size, shuffle=False
            )

        for j, batch in enumerate(tqdm(dataloader)):
            if test_cfg.cond == "unconditional":
                cond = None
                batch_size = batch
            else:
                cond = get_cond(
                    batch=batch,
                    tokenizer=model.tokenizer,
                    cond_type=test_cfg.cond,
                    model_type=type(model).__name__,
                )
                batch_size = cond["seq"].size(0)
                if multi_per_input:
                    batch_size = test_cfg.num_run

            t_start = time.time()
            layouts = model.sample(
                batch_size=batch_size,
                cond=cond,
                sampling_cfg=sampling_cfg,
                cond_type=test_cfg.cond,
            )
            t_end = time.time()
            t_total += t_end - t_start
            N_total += batch_size

            # visualize the results for sanity check, since the generation takes minutes to hours
            if j == 0:
                if not ckpt_dir.startswith("gs://"):
                    save_image(
                        layouts["bbox"],
                        layouts["label"],
                        layouts["mask"],
                        dataset.colors,
                        f"tmp/test_generated.png",
                    )

            if cond and "type" in cond and cond["type"] in ["partial", "refinement"]:
                if "bos" in model.tokenizer.special_tokens:
                    input_layouts = model.tokenizer.decode(cond["seq"][:, 1:].cpu())
                else:
                    is_diffusion = type(model).__name__ == "LayoutDM"
                    type_key = (
                        "seq_orig"
                        if cond["type"] == "refinement" and is_diffusion
                        else "seq"
                    )
                    input_layouts = model.tokenizer.decode(cond[type_key].cpu())
                inputs.extend(_filter_invalid(input_layouts))
            results.extend(_filter_invalid(layouts))

            # relation violation detection if necessary
            if test_cfg.cond == "relation":
                B = layouts["bbox"].size(0)
                canvas = torch.FloatTensor([0.5, 0.5, 1.0, 1.0])
                canvas_mask = torch.full((1,), fill_value=True)
                bbox_c = torch.cat(
                    [
                        repeat(canvas, "c -> b 1 c", b=B),
                        layouts["bbox"],
                    ],
                    dim=1,
                )
                mask_c = torch.cat(
                    [
                        repeat(canvas_mask, "1 -> b 1", b=B),
                        layouts["mask"],
                    ],
                    dim=1,
                )
                bbox_flatten = bbox_c[mask_c]
                if len(batch.edge_index) > 0:
                    v = compute_violation(bbox_flatten.to(device), batch)
                    v = v[~v.isnan()].sum().item()
                    batch_metrics["violation_score"] += v

                relation_scores.append(v)

        dummy_cfg = copy.deepcopy(train_cfg)
        dummy_cfg.sampling = sampling_cfg
        data = {"results": results, "train_cfg": dummy_cfg, "test_cfg": test_cfg}
        if len(inputs) > 0:
            data["inputs"] = inputs
        if len(relations) > 0:
            data["relations"] = relations
            data["relation_scores"] = relation_scores

        pkl_file = os.path.join(result_dir, f"seed_{seed_no}.pkl")
        with fs.open(pkl_file, "wb") as file_obj:
            pickle.dump(data, file_obj)

        print(N_total)
        print(f"ms per sample: {1e3 * t_total / N_total}")

        for k, v in batch_metrics.items():
            scores[k].append(v / len(results))

    keys, values = [], []
    for k, v in scores.items():
        v = np.array(v)
        mean, std = np.mean(v), np.std(v)
        keys += [f"{k}-mean", f"{k}-std"]
        values += [str(mean), str(std)]
    print(",".join(keys))
    print(",".join(values))


if __name__ == "__main__":
    filter_args_for_ai_platform()
    main()
