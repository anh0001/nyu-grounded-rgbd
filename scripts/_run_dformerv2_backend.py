"""Internal helper to run DFormerv2 inference and save raw label PNGs."""
from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


def build_dataset(config):
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.dataloader.dataloader import ValPre

    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }
    dataset = RGBXDataset(data_setting, "val", ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config))
    return dataset


def run_msf(model, images, modal_xs, config, device):
    _, _, h, w = images.shape
    scaled_logits = torch.zeros(images.shape[0], config.num_classes, h, w, device=device)
    scales = [float(s) for s in config.eval_scale_array]
    flip = bool(config.eval_flip)

    for scale in scales:
        new_h = int(np.ceil((scale * h) / 32.0) * 32)
        new_w = int(np.ceil((scale * w) / 32.0) * 32)
        scaled_images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=True)
        scaled_modal_xs = F.interpolate(modal_xs, size=(new_h, new_w), mode="bilinear", align_corners=True)
        logits = model(scaled_images, scaled_modal_xs)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True)
        scaled_logits += logits.softmax(dim=1)

        if flip:
            flip_images = torch.flip(scaled_images, dims=(3,))
            flip_modal_xs = torch.flip(scaled_modal_xs, dims=(3,))
            logits = model(flip_images, flip_modal_xs)
            logits = torch.flip(logits, dims=(3,))
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True)
            scaled_logits += logits.softmax(dim=1)

    return scaled_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--config-module", type=str, required=True)
    ap.add_argument("--export-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo_dir))

    cfg_mod = importlib.import_module(args.config_module)
    config = copy.deepcopy(cfg_mod.C)
    config.dataset_path = str(args.export_root)
    config.rgb_root_folder = str(args.export_root / "RGB")
    config.x_root_folder = str(args.export_root / "Depth")
    config.gt_root_folder = str(args.export_root / "Label")
    config.train_source = str(args.export_root / "train.txt")
    config.eval_source = str(args.export_root / "test.txt")
    config.rgb_format = ".png"
    config.x_format = ".png"
    config.gt_format = ".png"
    config.pad = False
    config.num_workers = 0

    from models.builder import EncoderDecoder
    from utils.load_utils import load_pretrain

    dataset = build_dataset(config)
    if args.limit is not None:
        dataset._file_names = dataset._file_names[: args.limit]

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    load_pretrain(model, str(args.checkpoint), strict=False)
    device = torch.device(args.device)
    model.to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for minibatch in loader:
            images = minibatch["data"].to(device)
            modal_xs = minibatch["modal_x"].to(device)
            logits = run_msf(model, images, modal_xs, config, device)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8) + 1
            stem = Path(minibatch["fn"][0]).stem
            Image.fromarray(pred, mode="L").save(args.out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
