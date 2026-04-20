"""Run zero-shot inference on a split and save semantic maps + reports."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("run_infer")


def load_cfg(path: Path):
    cfg = OmegaConf.load(path)
    # merge referenced sub-configs
    ds_cfg = OmegaConf.load(REPO / "configs" / "dataset" / f"{cfg.dataset}.yaml")
    mdl_cfg = OmegaConf.load(REPO / "configs" / "model" / f"{cfg.model}.yaml")
    pl_cfg = OmegaConf.load(REPO / "configs" / "pipeline" / f"{cfg.pipeline}.yaml")
    return cfg, ds_cfg, mdl_cfg, pl_cfg


def main() -> None:
    logger = setup_logging()
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
    report_dir = REPO / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    ds = NYUv2Dataset(root=REPO / ds_cfg.root, split=cfg.split, protocol=ds_cfg.protocol,
                      splits_file=ds_cfg.splits_file)
    logger.info("loaded config=%s experiment=%s split=%s device=%s", cfg_path, exp_name, cfg.split, args.device)
    logger.info("dataset=%s protocol=%s images=%d output_dir=%s", ds_cfg.name, ds_cfg.protocol, len(ds), out_dir)

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
    logger.info(
        "models ready gdino=%s sam=%s/%s pipeline=%s",
        mdl_cfg.gdino.model_id,
        mdl_cfg.sam.backend,
        mdl_cfg.sam.model_type,
        cfg.pipeline,
    )

    bank_file = OmegaConf.select(pl_cfg, "prompts.bank_file", default=None)
    chunks = build_chunks(
        protocol="nyu40" if ds_cfg.num_classes == 40 else "nyu13",
        bank_file=bank_file,
    )

    clip_reranker = None
    clip_cfg = OmegaConf.select(pl_cfg, "clip_rerank", default=None)
    if clip_cfg is not None and bool(OmegaConf.select(clip_cfg, "enabled", default=False)):
        from src.datasets.nyuv2_meta import NYU40_NAMES
        from src.models.clip_reranker import SigLIPReranker

        class_ids = list(range(1, 41))
        class_names = [NYU40_NAMES[i] for i in class_ids]
        clip_reranker = SigLIPReranker(
            model_id=str(OmegaConf.select(clip_cfg, "model_id",
                                          default="google/siglip-base-patch16-224")),
            device=args.device,
            class_names=class_names,
            class_ids=class_ids,
        )
        logger.info("clip reranker ready model=%s", clip_cfg.get("model_id", "siglip-base"))
    w_clip = float(OmegaConf.select(pl_cfg, "clip_rerank.w_clip", default=0.0))
    clip_reassign_margin = float(
        OmegaConf.select(pl_cfg, "clip_rerank.reassign_margin", default=0.0)
    )
    clip_reassign_min_top = float(
        OmegaConf.select(pl_cfg, "clip_rerank.reassign_min_top", default=0.0)
    )
    per_chunk_thresholds_cfg = OmegaConf.select(pl_cfg, "prompts.per_chunk_thresholds", default=None)
    per_chunk_thresholds: dict[str, tuple[float, float]] | None = None
    if per_chunk_thresholds_cfg is not None:
        per_chunk_thresholds = {
            str(k): (float(v[0]), float(v[1])) for k, v in dict(per_chunk_thresholds_cfg).items()
        }

    acc = ConfusionAccumulator(
        num_classes=ds_cfg.num_classes,
        ignore_index=ds_cfg.ignore_index,
        class_names=ds.class_names(),
    )

    limit = args.limit if args.limit is not None else cfg.limit
    ids = list(range(len(ds)))
    if limit:
        ids = ids[:int(limit)]
    logger.info("starting inference on %d image(s)", len(ids))

    flip_tta = bool(OmegaConf.select(pl_cfg, "fusion.flip_tta", default=False))
    cc_min_area = int(OmegaConf.select(pl_cfg, "fusion.cc_min_area", default=0))

    def _infer_frame(rgb, depth, valid):
        feat_local = compute_features(depth, valid=valid)
        cands_local = build_proposals(
            rgb=rgb, depth_feat=feat_local, gdino=gdino, sam=sam,
            chunks=chunks, image_id=None,
            box_threshold=float(mdl_cfg.gdino.box_threshold),
            text_threshold=float(mdl_cfg.gdino.text_threshold),
            box_iou_same=float(pl_cfg.box_nms.iou_same),
            box_iou_cross=float(pl_cfg.box_nms.iou_cross),
            mask_iou_same=float(pl_cfg.mask_nms.iou_same),
            mask_iou_cross=float(pl_cfg.mask_nms.iou_cross),
            per_chunk_thresholds=per_chunk_thresholds,
            clip_reranker=clip_reranker,
            w_clip=w_clip,
            clip_reassign_margin=clip_reassign_margin,
            clip_reassign_min_top=clip_reassign_min_top,
        )
        sem_local = rasterize(
            cands_local, feat_local, num_classes=40,
            fill_structural_by_geometry=bool(pl_cfg.fusion.fill_structural_by_geometry),
            cc_min_area=cc_min_area,
        )
        if bool(pl_cfg.fusion.residual_slic):
            sem_local = fill_residual_slic(sem_local, rgb, feat_local, cands_local)
        return sem_local

    t0 = time.time()
    for i in tqdm(ids, desc="infer", unit="img", dynamic_ncols=True):
        sample = ds[i]
        sem40 = _infer_frame(sample.rgb, sample.depth, sample.valid_depth)
        if flip_tta:
            import numpy as _np

            rgb_f = _np.ascontiguousarray(sample.rgb[:, ::-1])
            depth_f = _np.ascontiguousarray(sample.depth[:, ::-1])
            valid_f = (
                _np.ascontiguousarray(sample.valid_depth[:, ::-1])
                if sample.valid_depth is not None else None
            )
            sem_f = _infer_frame(rgb_f, depth_f, valid_f)[:, ::-1]
            # Majority vote between orig and flipped; tie → keep orig.
            disagree = sem40 != sem_f
            # For disagreement pixels, prefer the non-background/non-fallback class.
            from src.pipeline.semantic_fusion import (  # noqa: WPS433
                FALLBACK_PROP,
                FALLBACK_STUFF,
            )
            fb = {FALLBACK_PROP, FALLBACK_STUFF, 39}
            prefer_f = disagree & _np.isin(sem40, list(fb)) & ~_np.isin(sem_f, list(fb))
            sem40 = sem40.copy()
            sem40[prefer_f] = sem_f[prefer_f]
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
    with open(report_dir / f"{exp_name}.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "finished inference images=%d wall_time_s=%.2f report=%s",
        len(ids),
        report["wall_time_s"],
        report_dir / f"{exp_name}.json",
    )
    print(metrics.table())
    print(f"\nreport -> {report_dir / f'{exp_name}.json'}")


if __name__ == "__main__":
    main()
