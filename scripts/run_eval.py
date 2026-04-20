"""Evaluate saved prediction PNGs against NYUv2 ground truth."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf  # noqa: E402

from src.datasets.nyuv2 import NYUv2Dataset  # noqa: E402
from src.eval.metrics import ConfusionAccumulator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="nyuv2_40")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    ds_cfg = OmegaConf.load(REPO / "configs" / "dataset" / f"{args.dataset}.yaml")
    ds = NYUv2Dataset(root=REPO / ds_cfg.root, split=args.split,
                      protocol=ds_cfg.protocol, splits_file=ds_cfg.splits_file)
    acc = ConfusionAccumulator(num_classes=ds_cfg.num_classes,
                               ignore_index=ds_cfg.ignore_index,
                               class_names=ds.class_names())
    for i in tqdm(range(len(ds))):
        s = ds[i]
        p = args.pred_dir / f"{s.idx:04d}.png"
        if not p.exists():
            continue
        pred = np.array(Image.open(p))
        acc.update(pred, s.label)
    m = acc.result()
    print(m.table())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({
                "mean_iou": m.mean_iou,
                "pixel_acc": m.pixel_acc,
                "mean_class_acc": m.mean_class_acc,
                "per_class_iou": m.per_class_iou.tolist(),
            }, f, indent=2)


if __name__ == "__main__":
    main()
