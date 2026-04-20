"""NYUv2 dataset loader.

Loads the extracted layout from scripts/prepare_nyuv2.py:
  data/nyuv2/rgb/{idx:04d}.png
  data/nyuv2/depth/{idx:04d}.npy       float32 in-painted (H,W)
  data/nyuv2/depth_raw/{idx:04d}.npy   float32 raw (H,W), holes -> 0 or NaN
  data/nyuv2/labels40/{idx:04d}.png    uint8 (H,W), 0 = unlabeled
  data/nyuv2/labels13/{idx:04d}.png    uint8 (H,W), 0 = unlabeled
  data/nyuv2/splits/gupta_795_654.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


@dataclass
class NYUv2Sample:
    idx: int                       # 1-based image id
    rgb: np.ndarray                # (H,W,3) uint8
    depth: np.ndarray              # (H,W) float32, meters
    depth_raw: np.ndarray          # (H,W) float32, 0/NaN -> invalid
    label: np.ndarray              # (H,W) uint8, 0 = ignore
    valid_depth: np.ndarray        # (H,W) bool


class NYUv2Dataset:
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"] = "test",
        protocol: Literal["nyu40", "nyu13"] = "nyu40",
        splits_file: str = "splits/gupta_795_654.json",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.protocol = protocol
        self.label_subdir = "labels40" if protocol == "nyu40" else "labels13"
        self.num_classes = 40 if protocol == "nyu40" else 13

        with open(self.root / splits_file) as f:
            splits = json.load(f)
        self.ids: list[int] = list(splits[split])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> NYUv2Sample:
        idx = self.ids[i]
        rgb = np.array(Image.open(self.root / "rgb" / f"{idx:04d}.png").convert("RGB"))
        depth = np.load(self.root / "depth" / f"{idx:04d}.npy").astype(np.float32)
        depth_raw = np.load(self.root / "depth_raw" / f"{idx:04d}.npy").astype(np.float32)
        label = np.array(Image.open(self.root / self.label_subdir / f"{idx:04d}.png"))
        valid_depth = np.isfinite(depth_raw) & (depth_raw > 1e-3)
        return NYUv2Sample(
            idx=idx, rgb=rgb, depth=depth, depth_raw=depth_raw,
            label=label.astype(np.uint8), valid_depth=valid_depth,
        )

    def class_names(self) -> list[str]:
        from src.datasets.nyuv2_meta import NYU13_NAMES, NYU40_NAMES
        return NYU40_NAMES if self.protocol == "nyu40" else NYU13_NAMES
