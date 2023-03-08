import pathlib
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from einops import rearrange
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from torch import BoolTensor, FloatTensor, LongTensor

from .util import convert_xywh_to_ltrb


def convert_layout_to_image(
    boxes: FloatTensor,
    labels: LongTensor,
    colors: List[Tuple[int]],
    canvas_size: Optional[Tuple[int]] = (60, 40),
    resources: Optional[Dict] = None,
    names: Optional[List[str]] = None,
    **kwargs,
):
    H, W = canvas_size
    if names:
        font = ImageFont.truetype("LiberationSerif-Regular", W // 10)

    if resources:
        img = resources["img_bg"].resize((W, H))
    else:
        img = Image.new("RGB", (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # draw from larger boxes
    a = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(a)), key=lambda i: a[i], reverse=True)

    for i in indices:
        bbox, label = boxes[i], labels[i]
        if isinstance(label, LongTensor):
            label = label.item()

        c_fill = colors[label] + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)

        if resources:
            patch = resources["cropped_patches"][i]
            # round coordinates for exact size match for rendering images
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            w, h = x2 - x1, y2 - y1
            patch = patch.resize((w, h))
            img.paste(patch, (x1, y1))
        else:
            draw.rectangle([x1, y1, x2, y2], outline=colors[label], fill=c_fill)
            if names:
                # draw.text((x1, y1), names[label], colors[label], font=font)
                draw.text((x1, y1), names[label], "black", font=font)
    return img


def save_image(
    batch_boxes: FloatTensor,
    batch_labels: LongTensor,
    batch_mask: BoolTensor,
    colors: List[Tuple[int]],
    out_path: Optional[Union[pathlib.PosixPath, str]] = None,
    canvas_size: Optional[Tuple[int]] = (60, 40),
    nrow: Optional[int] = None,
    batch_resources: Optional[Dict] = None,
    use_grid: bool = False,
    draw_label: bool = False,
    **kwargs,
):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]
    if isinstance(out_path, pathlib.PosixPath):
        out_path = str(out_path)
    if not draw_label and "names" in kwargs:
        kwargs.pop("names")

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes = batch_boxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        if batch_resources:
            resources = {k: v[i] for (k, v) in batch_resources.items()}
            img = convert_layout_to_image(
                boxes, labels, colors, canvas_size, resources, **kwargs
            )
        else:
            img = convert_layout_to_image(boxes, labels, colors, canvas_size, **kwargs)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    if out_path:
        vutils.save_image(image, out_path, normalize=False, nrow=nrow)
    else:
        if use_grid:
            return torch_to_numpy_image(
                vutils.make_grid(image, normalize=False, nrow=nrow)
            )
        else:
            return image


def save_label(
    labels: Union[LongTensor, list],
    names: List[str],
    colors: List[Tuple[int]],
    out_path: Optional[Union[pathlib.PosixPath, str]] = None,
    **kwargs
    # canvas_size: Optional[Tuple[int]] = (60, 40),
):
    space, pad = 12, 12
    x_offset, y_offset = 500, 100

    img = Image.new("RGBA", (1000, 1000))
    fnt = ImageFont.truetype("LiberationSerif-Regular", 40)
    fnt_sm = ImageFont.truetype("LiberationSerif-Regular", 32)
    d = ImageDraw.Draw(img)

    if isinstance(labels, LongTensor):
        labels = labels.tolist()

    cnt = Counter(labels)
    for l in range(len(colors)):
        if l not in cnt.keys():
            continue

        text = names[l]
        use_multiline = False

        if cnt[l] > 1:
            add_width = d.textsize(f" × {cnt[l]}", font=fnt)[0] + pad
        else:
            add_width = 0

        width = d.textsize(text, font=fnt)[0]
        bbox = d.textbbox(
            (x_offset - (width + add_width) / 2, y_offset), text, font=fnt
        )
        bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        d.rectangle(bbox, fill=colors[l])

        _x_offset = x_offset - (width + add_width) / 2
        if cnt[l] > 1:
            d.text(
                (_x_offset + width + pad, y_offset),
                f" × {cnt[l]}",
                font=fnt,
                fill=(0, 0, 0),
            )

        d.text((_x_offset, y_offset), text, font=fnt, fill=(255, 255, 255))

        y_offset = y_offset + bbox[3] - bbox[1] + space

    # crop
    bbox = img.getbbox()
    bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    img = img.crop(bbox)

    # add white background
    out = Image.new("RGB", img.size, color=(255, 255, 255))
    out.paste(img, mask=img)
    # pil_size = canvas_size[::-1]  # (H, W) -> (W, H)
    # out = out.resize(pil_size)
    if out_path:
        out.save(out_path)
    else:
        return np.array(out)


def save_label_with_size(
    labels: LongTensor,
    boxes: FloatTensor,
    names: List[str],
    colors: List[Tuple[int]],
    out_path: Optional[Union[pathlib.PosixPath, str]] = None,
    # canvas_size: Optional[Tuple[int]] = (60, 40),
    **kwargs,
):
    space, pad = 12, 12
    x_offset, y_offset = 500, 100
    B = 32

    img = Image.new("RGBA", (1000, 1000))
    fnt = ImageFont.truetype("LiberationSerif-Regular", 40)
    fnt_sm = ImageFont.truetype("LiberationSerif-Regular", 32)
    d = ImageDraw.Draw(img)

    for i, l in enumerate(labels):
        w, h = [int(x) for x in (boxes[i].clip(1 / B, 1.0) * B).long()[2:]]
        text = f"{names[l]} ({w},{h})"
        add_width = 0

        width = d.textsize(text, font=fnt)[0]
        bbox = d.textbbox(
            (x_offset - (width + add_width) / 2, y_offset), text, font=fnt
        )
        bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        d.rectangle(bbox, fill=colors[l])

        _x_offset = x_offset - (width + add_width) / 2
        d.text((_x_offset, y_offset), text, font=fnt, fill=(255, 255, 255))
        y_offset = y_offset + bbox[3] - bbox[1] + space

    # crop
    bbox = img.getbbox()
    bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    img = img.crop(bbox)

    # add white background
    out = Image.new("RGB", img.size, color=(255, 255, 255))
    out.paste(img, mask=img)
    # pil_size = canvas_size[::-1]  # (H, W) -> (W, H)
    # out = out.resize(pil_size)
    if out_path:
        out.save(out_path)
    else:
        return np.array(out)


def torch_to_numpy_image(input_th: FloatTensor) -> np.ndarray:
    """
    Args
        input_th: (C, H, W), in [0.0, 1/0], torch image
    Returns
        output_npy: (H, W, C), in {0, 1, ..., 255}, numpy image
    """
    x = (input_th * 255.0).clamp(0, 255)
    x = rearrange(x, "c h w -> h w c")
    output_npy = x.numpy().astype(np.uint8)
    return output_npy


def save_relation(
    label_with_canvas: LongTensor,
    edge_attr: LongTensor,
    names: List[str],
    colors: List[Tuple[int]],
    out_path: Optional[Union[pathlib.PosixPath, str]] = None,
    **kwargs,
):
    from trainer.data.util import (  # lazy load to avoid circular import
        RelLoc,
        RelSize,
        get_rel_text,
    )

    pairs, triplets = [], []
    relations = list(RelSize) + list(RelLoc)
    for rel_value in relations:
        if rel_value in [RelSize.UNKNOWN, RelLoc.UNKNOWN]:
            continue
        cond = edge_attr & 1 << rel_value
        ii, jj = np.where(cond.numpy() > 0)
        for i, j in zip(ii, jj):
            li = label_with_canvas[i] - 1
            lj = label_with_canvas[j] - 1

            if i == 0:
                rel = get_rel_text(rel_value, canvas=True)
                pairs.append((lj, rel, None))
            else:
                rel = get_rel_text(rel_value, canvas=False)
                triplets.append((li, rel, lj))

    triplets = pairs + triplets

    space, pad = 6, 6
    img = Image.new("RGBA", (1000, 1000))
    fnt = ImageFont.truetype("LiberationSerif-Regular", 20)
    fnt_sm = ImageFont.truetype("LiberationSerif-Regular", 16)
    d = ImageDraw.Draw(img)

    def draw_text(x_offset, y_offset, text, color=None, first=False):
        if color is None:
            d.text((x_offset, y_offset), text, font=fnt, fill=(0, 0, 0))
            x_offset = x_offset + d.textsize(text, font=fnt)[0] + space
        else:
            x_offset = x_offset + pad

            use_multiline = False
            bbox = d.textbbox((x_offset, y_offset), text, font=fnt)
            if bbox[2] - bbox[0] > 120 and " " in text:
                use_multiline = True
                h_old = d.textsize(text, font=fnt)[1]
                text = text.replace(" ", "\n")
                h_new = d.multiline_textsize(text, font=fnt_sm)[1]
                h_diff = h_new - h_old
                if first:
                    y_offset = y_offset + h_diff / 2
                bbox = d.multiline_textbbox(
                    (x_offset, y_offset - h_diff / 2), text, font=fnt_sm
                )

            bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
            d.rectangle(bbox, fill=color)

            if use_multiline:
                d.multiline_text(
                    (x_offset, y_offset - h_diff / 2),
                    text,
                    align="center",
                    font=fnt_sm,
                    fill=(255, 255, 255),
                )
                text_width = d.multiline_textsize(text, font=fnt_sm)[0]
            else:
                d.text((x_offset, y_offset), text, font=fnt, fill=(255, 255, 255))
                text_width = d.textsize(text, font=fnt)[0]

            x_offset = x_offset + text_width + space + pad
        return x_offset, y_offset

    for i, (l1, rel, l2) in enumerate(triplets):
        x_offset, y_offset = 20, 40 * (i + 1)
        x_offset, y_offset = draw_text(
            x_offset, y_offset, names[l1], colors[l1], first=True
        )
        x_offset, y_offset = draw_text(x_offset, y_offset, rel)
        if l2 is not None:
            draw_text(x_offset, y_offset, names[l2], colors[l2])

    # crop
    bbox = img.getbbox()
    if bbox is not None:
        bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        img = img.crop(bbox)

        # add white background
        out = Image.new("RGB", img.size, color=(255, 255, 255))
        out.paste(img, mask=img)

        if out_path:
            out.save(out_path)
        else:
            return np.array(out)


def save_gif(
    images: List[NDArray],
    out_path: str,
    **kwargs,
):
    assert images[0].ndim == 4
    to_pil = T.ToPILImage()
    for i in range(len(images[0])):
        tmp_images = [to_pil(image[i]) for image in images]
        tmp_images[0].save(
            # f"tmp/animation/{i}.gif",
            out_path.format(i),
            save_all=True,
            append_images=tmp_images[1:],
            optimize=False,
            duration=200,
            loop=0,
        )
