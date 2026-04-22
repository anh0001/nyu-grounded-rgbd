"""Internal helper to run ESANet inference and save raw label PNGs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--export-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo_dir))

    from src.build_model import build_model
    from src.datasets.nyuv2.pytorch_dataset import NYUv2
    from src.preprocessing import get_preprocessor

    class Args:
        pass

    esanet_args = Args()
    esanet_args.dataset = "nyuv2"
    esanet_args.pretrained_on_imagenet = False
    esanet_args.height = int(args.height)
    esanet_args.width = int(args.width)
    esanet_args.valid_full_res = False
    esanet_args.batch_size = 1
    esanet_args.batch_size_valid = 1
    esanet_args.workers = 0
    esanet_args.last_ckpt = ""
    esanet_args.pretrained_scenenet = ""
    esanet_args.modality = "rgbd"
    esanet_args.pretrained_dir = str(args.repo_dir / "trained_models" / "imagenet")
    esanet_args.encoder = "resnet34"
    esanet_args.encoder_block = "NonBottleneck1D"
    esanet_args.nr_decoder_blocks = [3]
    esanet_args.encoder_depth = None
    esanet_args.activation = "relu"
    esanet_args.encoder_decoder_fusion = "add"
    esanet_args.context_module = "ppm"
    esanet_args.channels_decoder = 128
    esanet_args.decoder_channels_mode = "decreasing"
    esanet_args.fuse_depth_in_rgb_encoder = "SE-add"
    esanet_args.upsampling = "learned-3x3-zeropad"
    esanet_args.he_init = False
    esanet_args.finetune = None
    esanet_args.freeze = 0

    dataset = NYUv2(
        data_dir=str(args.export_root),
        n_classes=40,
        split="test",
        depth_mode="refined",
        with_input_orig=True,
    )
    dataset.preprocessor = get_preprocessor(
        height=esanet_args.height,
        width=esanet_args.width,
        depth_mean=dataset.depth_mean,
        depth_std=dataset.depth_std,
        depth_mode="refined",
        phase="test",
    )
    if args.limit is not None:
        dataset._filenames = dataset._filenames[: args.limit]
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    n_classes = dataset.n_classes_without_void
    model, _ = build_model(esanet_args, n_classes=n_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data_loader, unit="img")):
            image = sample["image"].to(device)
            depth = sample["depth"].to(device)
            label_orig = sample["label_orig"]
            _, image_h, image_w = label_orig.shape

            pred = model(image, depth)
            pred = F.interpolate(pred, (image_h, image_w), mode="bilinear", align_corners=False)
            pred = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8) + 1
            stem = data_loader.dataset._filenames[idx]
            Image.fromarray(pred, mode="L").save(args.out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
