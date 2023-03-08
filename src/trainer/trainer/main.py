import json
import logging
import os
import time

import hydra
import torch
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader  # noqa
from trainer.data.util import compose_transform, sparse_to_dense, split_num_samples
from trainer.fid.model import load_fidnet_v3
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.metric import compute_generative_model_scores
from trainer.helpers.sampling import register_sampling_config
from trainer.helpers.scheduler import ReduceLROnPlateauWithWarmup
from trainer.helpers.util import set_seed
from trainer.helpers.visualization import save_image
from trainer.hydra_configs import DataConfig, TrainConfig
from trainer.models.common.util import save_model

from .crossplatform_util import filter_args_for_ai_platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra

torch.autograd.set_detect_anomaly(True)
total_iter_count = 0


def _to(inputs, device):
    """
    recursively send tensor to the specified device
    """
    outputs = {}
    for k, v in inputs.items():
        if isinstance(v, dict):
            outputs[k] = _to(v, device)
        elif isinstance(v, Tensor):
            outputs[k] = v.to(device)
    return outputs


# if config is not used by hydra.utils.instantiate, define schema to validate args
cs = ConfigStore.instance()
cs.store(group="data", name="base_data_default", node=DataConfig)
cs.store(group="training", name="base_training_default", node=TrainConfig)
register_sampling_config(cs)


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    global total_iter_count
    job_dir = os.path.join(cfg.job_dir, str(cfg.seed))

    fs, _ = url_to_fs(job_dir)
    if not fs.exists(job_dir):
        fs.mkdir(job_dir)
    writer = SummaryWriter(os.path.join(job_dir, "logs"))
    logger.info(cfg)

    if cfg.debug:
        cfg.data.num_workers = 1
        cfg.training.epochs = 2
        cfg.data.batch_size = 64

    with fs.open(os.path.join(job_dir, "config.yaml"), "wb") as file_obj:
        file_obj.write(OmegaConf.to_yaml(cfg).encode("utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = compose_transform(cfg.data.transforms)
    train_dataset = instantiate(cfg.dataset)(split="train", transform=transform)
    val_dataset = instantiate(cfg.dataset)(split="val", transform=transform)

    kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **kwargs)

    tokenizer = LayoutSequenceTokenizer(data_cfg=cfg.data, dataset_cfg=cfg.dataset)
    model = instantiate(cfg.model)(backbone_cfg=cfg.backbone, tokenizer=tokenizer)
    model = model.to(device)

    optim_groups = model.optim_groups(cfg.training.weight_decay)
    optimizer = instantiate(cfg.optimizer)(optim_groups)
    scheduler = instantiate(cfg.scheduler)(optimizer=optimizer)

    fid_model = load_fidnet_v3(train_dataset, cfg.fid_weight_dir, device)

    best_val_loss = float("Inf")
    for epoch in range(cfg.training.epochs):
        model.update_per_epoch(epoch, cfg.training.epochs)

        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, cfg, device, writer)
        val_loss = evaluate(model, val_dataloader, cfg, device)
        logger.info(
            "Epoch %d: elapsed = %.1fs, train_loss = %.4f, val_loss = %.4f"
            % (epoch + 1, time.time() - start_time, train_loss, val_loss)
        )
        if any(
            isinstance(scheduler, s)
            for s in [ReduceLROnPlateau, ReduceLROnPlateauWithWarmup]
        ):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.add_scalar("train_loss_epoch_avg", train_loss, epoch + 1)
        writer.add_scalar("val_loss_epoch_avg", val_loss, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, job_dir, best_or_final="best")

        if (epoch + 1) % cfg.training.sample_plot_epoch_interval == 0:
            with torch.set_grad_enabled(False):
                layouts = model.sample(
                    batch_size=cfg.data.batch_size,
                    sampling_cfg=cfg.sampling,
                    device=device,
                )
            images = save_image(
                layouts["bbox"],
                layouts["label"],
                layouts["mask"],
                val_dataset.colors,
            )
            tag = f"{cfg.sampling.name} sampling results"
            writer.add_images(tag, images, epoch + 1)

            if cfg.debug:
                save_image(
                    layouts["bbox"],
                    layouts["label"],
                    layouts["mask"],
                    val_dataset.colors,
                    f"tmp/debug_{total_iter_count}.png",
                )

        fid_epoch_interval = 1 if cfg.debug else cfg.training.epochs // 10
        # fid_epoch_interval = 3 if cfg.debug else cfg.training.epochs // 10

        if (epoch + 1) % fid_epoch_interval == 0:
            N = cfg.training.fid_plot_num_samples
            val_dataloader_fid = DataLoader(val_dataset[:N], shuffle=False, **kwargs)
            feats_1, feats_2 = [], []

            for batch in val_dataloader_fid:
                remove_canvas = (
                    cfg.model._target_ == "trainer.models.unilayout.UniLayout"
                )
                bbox, label, padding_mask, _ = sparse_to_dense(
                    batch, device, remove_canvas=remove_canvas
                )
                with torch.set_grad_enabled(False):
                    feat = fid_model.extract_features(bbox, label, padding_mask)
                feats_1.append(feat.cpu())

            dataloader = split_num_samples(
                cfg.training.fid_plot_num_samples, cfg.training.fid_plot_batch_size
            )
            for batch_size in dataloader:
                with torch.set_grad_enabled(False):
                    layouts = model.sample(
                        batch_size=batch_size,
                        sampling_cfg=cfg.sampling,
                        device=device,
                    )
                    feat = fid_model.extract_features(
                        layouts["bbox"].to(device),
                        layouts["label"].to(device),
                        ~layouts["mask"].to(device),
                    )

                feats_2.append(feat.cpu())
            fid_results = compute_generative_model_scores(feats_1, feats_2)
            if cfg.debug:
                print(fid_results)
            for k, v in fid_results.items():
                writer.add_scalar(f"val_{k}", v, epoch + 1)

    test_dataset = instantiate(cfg.dataset)(split="test", transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **kwargs)
    test_loss = evaluate(model, test_dataloader, cfg, device)
    logger.info("test_loss = %.4f" % (test_loss))
    result = {"test_loss": test_loss}

    # Save results and model weights.
    with fs.open(os.path.join(job_dir, "result.json"), "wb") as file_obj:
        file_obj.write(json.dumps(result).encode("utf-8"))
    save_model(model, job_dir, best_or_final="final")
    total_iter_count = 0  # reset iter count for multirun


def train(
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    writer: SummaryWriter,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    global total_iter_count

    for batch in train_data:
        batch = model.preprocess(batch)
        batch = _to(batch, device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, losses = model(batch)
            loss = sum(losses.values())
        loss.backward()  # type: ignore

        if cfg.training.grad_norm_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_norm_clip
            )

        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
        total_iter_count += 1

        if total_iter_count % cfg.training.loss_plot_iter_interval == 0:
            for (k, v) in losses.items():
                writer.add_scalar(k, v.cpu().item(), total_iter_count + 1)

        # below are for development

        # if cfg.debug:
        #     break

        # if cfg.debug and total_iter_count % 10 == 0:
        #     text = ""
        #     for (k, v) in losses.items():
        #         text += f"{k}: {v} "
        #     print(total_iter_count, text)

        # if cfg.debug and total_iter_count % (cfg.training.loss_plot_iter_interval * 10) == 0:
        #     # sanity check
        #     if cfg.debug:
        #         layouts = model.tokenizer.decode(outputs["outputs"].cpu())
        #         save_image(
        #             layouts["bbox"],
        #             layouts["label"],
        #             layouts["mask"],
        #             train_data.dataset.colors,
        #             f"tmp/debug_{total_iter_count}.png",
        #         )

    return total_loss / steps


def evaluate(
    model: torch.nn.Module,
    test_data: DataLoader,
    cfg: DictConfig,
    device: torch.device,
) -> float:
    total_loss = 0.0
    steps = 0

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in test_data:
            batch = model.preprocess(batch)
            # batch = {k: v.to(device) for (k, v) in batch.items()}
            batch = _to(batch, device)
            _, losses = model(batch)
            loss = sum(losses.values())
            total_loss += float(loss.item())
            steps += 1

            if cfg.debug:
                break

    return total_loss / steps


if __name__ == "__main__":
    filter_args_for_ai_platform()
    main()
