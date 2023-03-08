import argparse
import os
import shutil
from pathlib import Path

import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from trainer.data.util import AddNoiseToBBox, LexicographicOrder
from trainer.fid.model import FIDNetV3
from trainer.global_configs import DATASET_DIR
from trainer.helpers.visualization import save_image


def save_checkpoint(state, is_best, out_dir):
    out_path = Path(out_dir) / "checkpoint.pth.tar"
    torch.save(state, out_path)

    if is_best:
        best_path = Path(out_dir) / "model_best.pth.tar"
        shutil.copyfile(out_path, best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_yaml", type=str)
    parser.add_argument("--out_dir", type=str, default="tmp/fid_weights")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--iteration",
        type=int,
        default=int(2e5),
        help="number of iterations to train for",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="learning rate, default=3e-4"
    )
    parser.add_argument("--seed", type=int, help="manual seed")
    args = parser.parse_args()
    print(args)

    dataset_cfg = omegaconf.OmegaConf.load(args.dataset_yaml)
    dataset_cfg["dir"] = DATASET_DIR

    prefix = "FIDNetV3"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "logs"))

    transform = T.Compose(
        [
            T.RandomApply([AddNoiseToBBox()], 0.5),
            LexicographicOrder(),
        ]
    )
    train_dataset = instantiate(dataset_cfg)(split="train", transform=transform)
    val_dataset = instantiate(dataset_cfg)(split="test", transform=transform)
    categories = train_dataset.labels

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": os.cpu_count(),
        "pin_memory": True,
    }

    train_dataloader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **kwargs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FIDNetV3(num_label=len(categories), max_bbox=dataset_cfg.max_seq_length).to(
        device
    )

    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
    criterion_label = nn.CrossEntropyLoss(reduction="none")
    criterion_bbox = nn.MSELoss(reduction="none")

    def proc_batch(batch):
        batch = batch.to(device)
        bbox, _ = to_dense_batch(batch.x, batch.batch)
        label, mask = to_dense_batch(batch.y, batch.batch)
        padding_mask = ~mask

        is_real = batch.attr["NoiseAdded"].float()
        return bbox, label, padding_mask, mask, is_real

    iteration = 0
    best_loss = 1e8
    max_epoch = args.iteration * args.batch_size / len(train_dataset)
    max_epoch = torch.ceil(torch.tensor(max_epoch)).int().item()
    for epoch in range(max_epoch):
        model.train()
        train_loss = {
            "Loss_BCE": 0,
            "Loss_Label": 0,
            "Loss_BBox": 0,
        }

        for i, batch in enumerate(train_dataloader):
            bbox, label, padding_mask, mask, is_real = proc_batch(batch)
            model.zero_grad()

            logit, logit_cls, bbox_pred = model(bbox, label, padding_mask)

            loss_bce = criterion_bce(logit, is_real)
            loss_label = criterion_label(logit_cls[mask], label[mask])
            loss_bbox = criterion_bbox(bbox_pred[mask], bbox[mask]).sum(-1)
            loss = loss_bce.mean() + loss_label.mean() + 10 * loss_bbox.mean()
            loss.backward()

            optimizer.step()

            loss_bce_mean = loss_bce.mean().item()
            train_loss["Loss_BCE"] += loss_bce.sum().item()
            loss_label_mean = loss_label.mean().item()
            train_loss["Loss_Label"] += loss_label.sum().item()
            loss_bbox_mean = loss_bbox.mean().item()
            train_loss["Loss_BBox"] += loss_bbox.sum().item()

            # add data to tensorboard
            writer.add_scalar(prefix + "/Loss", loss.item(), iteration)
            writer.add_scalar(prefix + "/Loss_BCE", loss_bce_mean, iteration)
            writer.add_scalar(prefix + "/Loss_Label", loss_label_mean, iteration)
            writer.add_scalar(prefix + "/Loss_BBox", loss_bbox_mean, iteration)

            if i % 50 == 0:
                log_prefix = f"[{epoch}/{max_epoch}][{i}/{len(train_dataset) // args.batch_size}]"
                log = f"Loss: {loss.item():E}\tBCE: {loss_bce_mean:E}\tLabel: {loss_label_mean:E}\tBBox: {loss_bbox_mean:E}"
                print(f"{log_prefix}\t{log}")

            iteration += 1

        for key in train_loss.keys():
            train_loss[key] /= len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_loss = {
                "Loss_BCE": 0,
                "Loss_Label": 0,
                "Loss_BBox": 0,
            }

            for i, batch in enumerate(val_dataloader):
                bbox, label, padding_mask, mask, is_real = proc_batch(batch)

                logit, logit_cls, bbox_pred = model(bbox, label, padding_mask)

                loss_bce = criterion_bce(logit, is_real)
                loss_label = criterion_label(logit_cls[mask], label[mask])
                loss_bbox = criterion_bbox(bbox_pred[mask], bbox[mask]).sum(-1)

                val_loss["Loss_BCE"] += loss_bce.sum().item()
                val_loss["Loss_Label"] += loss_label.sum().item()
                val_loss["Loss_BBox"] += loss_bbox.sum().item()

                if i == 0 and epoch % 10 == 0:
                    save_image(
                        bbox,
                        label,
                        mask,
                        val_dataset.colors,
                        out_dir / f"samples_{epoch}.png",
                    )
                    cls_pred = logit_cls.argmax(dim=-1)
                    save_image(
                        bbox_pred,
                        cls_pred,
                        mask,
                        val_dataset.colors,
                        out_dir / f"recon_samples_{epoch}.png",
                    )

            for key in val_loss.keys():
                val_loss[key] /= len(val_dataset)

        writer.add_scalar(prefix + "/Epoch", epoch, iteration)
        tag_scalar_dict = {
            "train": sum(train_loss.values()),
            "val": sum(val_loss.values()),
        }
        writer.add_scalars(prefix + "/Loss_Epoch", tag_scalar_dict, iteration)
        for key in train_loss.keys():
            tag_scalar_dict = {"train": train_loss[key], "val": val_loss[key]}
            writer.add_scalars(prefix + f"/{key}_Epoch", tag_scalar_dict, iteration)

        # do checkpointing
        val_loss = sum(val_loss.values())
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            out_dir,
        )


if __name__ == "__main__":
    main()
