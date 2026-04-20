"""Run zero-shot inference on a split and save semantic maps + reports."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets.nyuv2 import NYUv2Dataset  # noqa: E402
from src.eval.metrics import ConfusionAccumulator  # noqa: E402
from src.models.gdino import GroundingDINO  # noqa: E402
from src.models.sam_wrapper import SAMWrapper  # noqa: E402
from src.pipeline.depth_features import compute_features  # noqa: E402
from src.pipeline.proposals import build_proposals  # noqa: E402
from src.pipeline.region_fill import fill_residual_slic  # noqa: E402
from src.pipeline.semantic_fusion import rasterize  # noqa: E402
from src.prompts.alias_bank import build_chunks  # noqa: E402


def load_cfg(path: Path):
    cfg = OmegaConf.load(path)
    # merge referenced sub-configs
    ds_cfg = OmegaConf.load(REPO / "configs" / "dataset" / f"{cfg.dataset}.yaml")
    mdl_cfg = OmegaConf.load(REPO / "configs" / "model" / f"{cfg.model}.yaml")
    pl_cfg = OmegaConf.load(REPO / "configs" / "pipeline" / f"{cfg.pipeline}.yaml")
    return cfg, ds_cfg, mdl_cfg, pl_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="experiment config name (no .yaml) or path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = REPO / "configs" / "experiment" / f"{args.config}.yaml"
    cfg, ds_cfg, mdl_cfg, pl_cfg = load_cfg(cfg_path)

    exp_name = cfg.name
    out_dir = args.out or (REPO / "outputs" / "predictions" / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = NYUv2Dataset(root=REPO / ds_cfg.root, split=cfg.split, protocol=ds_cfg.protocol,
                      splits_file=ds_cfg.splits_file)

    gdino = GroundingDINO(
        model_id=mdl_cfg.gdino.model_id,
        device=args.device,
        box_threshold=float(mdl_cfg.gdino.box_threshold),
        text_threshold=float(mdl_cfg.gdino.text_threshold),
    )
    sam = SAMWrapper(
        backend=mdl_cfg.sam.backend,
        checkpoint=mdl_cfg.sam.checkpoint,
        model_type=mdl_cfg.sam.model_type,
        device=args.device,
    )

    chunks = build_chunks(protocol="nyu40" if ds_cfg.num_classes == 40 else "nyu13")

    acc = ConfusionAccumulator(
        num_classes=ds_cfg.num_classes,
        ignore_index=ds_cfg.ignore_index,
        class_names=ds.class_names(),
    )

    limit = args.limit if args.limit is not None else cfg.limit
    ids = list(range(len(ds)))
    if limit:
        ids = ids[:int(limit)]

    t0 = time.time()
    for i in tqdm(ids):
        sample = ds[i]
        feat = compute_features(sample.depth, valid=sample.valid_depth)
        cands = build_proposals(
            rgb=sample.rgb, depth_feat=feat, gdino=gdino, sam=sam,
            chunks=chunks, image_id=f"{sample.idx:04d}",
            box_threshold=float(mdl_cfg.gdino.box_threshold),
            text_threshold=float(mdl_cfg.gdino.text_threshold),
            box_iou_same=float(pl_cfg.box_nms.iou_same),
            box_iou_cross=float(pl_cfg.box_nms.iou_cross),
            mask_iou_same=float(pl_cfg.mask_nms.iou_same),
            mask_iou_cross=float(pl_cfg.mask_nms.iou_cross),
        )
        sem40 = rasterize(
            cands, feat, num_classes=40,
            fill_structural_by_geometry=bool(pl_cfg.fusion.fill_structural_by_geometry),
        )
        if bool(pl_cfg.fusion.residual_slic):
            sem40 = fill_residual_slic(sem40, sample.rgb, feat, cands)
        if ds_cfg.num_classes == 13:
            from src.pipeline.semantic_fusion import _map_40_to_13
            pred = _map_40_to_13(sem40)
        else:
            pred = sem40
        Image.fromarray(pred).save(out_dir / f"{sample.idx:04d}.png")
        acc.update(pred, sample.label)

    metrics = acc.result()
    report = {
        "experiment": exp_name,
        "dataset": ds_cfg.name,
        "num_images": len(ids),
        "mean_iou": metrics.mean_iou,
        "pixel_acc": metrics.pixel_acc,
        "mean_class_acc": metrics.mean_class_acc,
        "per_class_iou": {
            (metrics.class_names[i + 1] if i + 1 < len(metrics.class_names) else f"c{i+1}"):
                float(metrics.per_class_iou[i])
            for i in range(metrics.num_classes)
        },
        "wall_time_s": time.time() - t0,
    }
    report_dir = REPO / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"{exp_name}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(metrics.table())
    print(f"\nreport -> {report_dir / f'{exp_name}.json'}")


if __name__ == "__main__":
    main()
